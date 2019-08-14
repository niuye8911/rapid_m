from RapidProfile import RapidProfile

testDF = RapidProfile('../testData/facedetect-perf.csv')
testDF.scale()
testDF.writeOut('tmp.csv')
