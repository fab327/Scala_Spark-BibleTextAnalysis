import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

object RunLSA {
  def main(args: Array[String]) {

    val k = if (args.length > 0) args(0).toInt else 100
    val numTerms = if (args.length > 1) args(1).toInt else 50000
    val sampleSize = if (args.length > 2) args(2).toDouble else 0.01

    val conf = new SparkConf()
      .setAppName("BibleLSA")
      .setMaster("local[*]")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.executor.memory", "6g")
    val sc = new SparkContext(conf)

    val (termDocMatrix, termIds, docIds, idfs) = preprocessing(sampleSize, numTerms, sc)

    //REQUIREMENT #3: Conduct Singular Value Decomposition
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #3: Conduct Singular Value Decomposition")
    termDocMatrix.cache()
    val mat = new RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(k, computeU = true)
    println("Singular values: " + svd.s)

    //REQUIREMENT #4: Finding Important Concepts
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #4: Finding Important Concepts")
    val topConceptTerms = topTermsInTopConcepts(svd, 10, 10, termIds)
    val topConceptDocs = topDocsInTopConcepts(svd, 10, 10, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      println("Concept docs: " + docs.map(_._1).mkString(", "))
      println()
    }

    //REQUIREMENT #5: Finding Term-Term Relevance
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #5: Finding Term-Term Relevance")
    val VS = multiplyByDiagonalMatrix(svd.V, svd.s)
    val normalizedVS = rowsNormalized(VS)
    val idTerms = idfs.keys.zipWithIndex.toMap
    printTopTermsForTerm(normalizedVS, "child", Map("child" -> idTerms("child")), termIds)

    //REQUIREMENT #6: Finding Verse-Verse Relevance
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #6: Finding Verse-Verse Relevance")
    val US = multiplyByDiagonalMatrix(svd.U, svd.s)
    val normalizedUS = rowsNormalized(US)
    val idDocs = docIds.keys.zipWithIndex.toMap
    printTopDocsForDoc(normalizedUS, "Ne 7:56", Map("Ne 7:56" -> idDocs(5)), docIds)

    //REQUIREMENT #7: Finding Term-Verse Relevance
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #7: Finding Term-Verse Relevance")
    printTopDocsForTerm(normalizedUS, svd.V, "child", Map("child" -> idTerms("child")) ,docIds)

    //REQUIREMENT #8: Conducting multi-term queries
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #8: Conducting multi-term queries")
    printRelevantDocs(Seq("child", "God"), idTerms, idfs, US, svd.V, docIds)
  }

  /**
    * Returns an RDD of rows of the term document matrix, a mapping of column indices to terms, and a
    * mapping of row IDs to document titles.
    */
  def preprocessing(sampleSize: Double, numTerms: Int, sc: SparkContext)
  : (RDD[Vector], Map[Int, String], Map[Long, String], Map[String, Double]) = {

    var BibleRDD = sc.textFile(".\\resources\\KJV.txt")
    if (determineOS().equals("Mac")) {
      BibleRDD = sc.textFile("./resources/KJV.txt")
    }
    val verses = BibleRDD.filter(line => isVerse(line)).map(a => a.substring(3))
    val contents = BibleRDD.filter(line => !isVerse(line))
    val versesContent = verses.zip(contents)

    var stopWordsPath = ".\\resources\\stopwords.txt"
    if (determineOS().equals("Mac")) {
      stopWordsPath = "./resources/stopwords.txt"
    }
    val stopWords = sc.broadcast(ParseBible.loadStopWords(stopWordsPath)).value


    //REQUIREMENT #1: Lemmatize the whole collection
    println("------------------------------------------------------------------------------------------")
    println("REQUIREMENT #1: Lemmatize the whole collection")
    val lemmatized = versesContent.mapPartitions(iter => {
      val pipeline = ParseBible.createNLPPipeline()
      iter.map { case (title, contents) => (title, ParseBible.plainTextToLemmas(contents, stopWords, pipeline)) }
    })

    val filtered = lemmatized.filter(_._2.size > 1)

    ParseBible.termDocumentMatrix(filtered, stopWords, numTerms, sc)
  }

  def isContent(line: String) = !isVerse(line)
  def isVerse(line: String) = line.contains("$$")

  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                            numTerms: Int, termIds: Map[Int, String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map { case (score, id) => (termIds(id), score) }
    }
    topTerms
  }

  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                           numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId
      topDocs += docWeights.top(numDocs).map { case (score, id) => (docIds(id), score) }
    }
    topDocs
  }

  def printTopTermsForTerm(normalizedVS: BDenseMatrix[Double],
                           term: String, idTerms: Map[String, Int], termIds: Map[Int, String]) {
    printIdWeights(topTermsForTerm(normalizedVS, idTerms(term)), termIds)
  }

  /**
    * Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
    * relevance scores to the given term.
    */
  def topTermsForTerm(normalizedVS: BDenseMatrix[Double], termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val termRowVec = new BDenseVector[Double](row(normalizedVS, termId).toArray)

    // Compute scores against every term
    val termScores = (normalizedVS * termRowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  /**
    * Selects a row from a matrix.
    */
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = {
    (0 until mat.cols).map(c => mat(index, c))
  }

  def printTopDocsForDoc(normalizedUS: RowMatrix, doc: String, idDocs: Map[String, Long],
                         docIds: Map[Long, String]) {
    printIdWeights(topDocsForDoc(normalizedUS, idDocs(doc)), docIds)
  }

  /**
    * Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given doc.
    */
  def topDocsForDoc(normalizedUS: RowMatrix, docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = row(normalizedUS, docId)
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    // Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /**
    * Selects a row from a distributed matrix.
    */
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId.map(_.swap).lookup(id).head.toArray
  }

  def printTopDocsForTerm(US: RowMatrix, V: Matrix, term: String, idTerms: Map[String, Int],
                          docIds: Map[Long, String]) {
    printIdWeights(topDocsForTerm(US, V, idTerms(term)), docIds)
  }

  /**
    * Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given term.
    */
  def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int): Seq[(Double, Long)] = {
    val termRowArr = row(V, termId).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printRelevantDocs(terms: Seq[String], idTerms: Map[String, Int], idfs: Map[String, Double],
                        US: RowMatrix, V: Matrix, docIds: Map[Long, String]): Unit = {
    val queryVec = termsToQueryVector(terms, idTerms, idfs);
    printIdWeights(topDocsForTermQuery(US, V, queryVec), docIds);
  }

  def termsToQueryVector(terms: Seq[String], idTerms: Map[String, Int], idfs: Map[String, Double])
  : BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = terms.map(idfs(_)).toArray
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  def topDocsForTermQuery(US: RowMatrix, V: Matrix, query: BSparseVector[Double])
  : Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  /**
    * Selects a row from a matrix.
    */
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    (0 until mat.numCols).map(i => arr(index + i * mat.numRows))
  }

  def printIdWeights[T](idWeights: Seq[(Double, T)], entityIds: Map[T, String]) {
    println(idWeights.map { case (score, id) => (entityIds(id), score) }.mkString(", "))
  }

  /**
    * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
    * Breeze doesn't support efficient diagonal representations, so multiply manually.
    */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs{case ((r, c), v) => v * sArr(c)}
  }

  /**
    * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
    */
  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    }))
  }

  /**
    * Returns a matrix where each row is divided by its length.
    */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).map(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /**
    * Returns a distributed matrix where each row is divided by its length.
    */
  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

  def determineOS(): String = {
    val operatingSystem = System.getProperty("os.name");
    if (operatingSystem.contains("Windows"))
      "Windows";
    else if (operatingSystem.contains("Mac"))
      "Mac";
    else
      "";
  }

}
