export CLASSPATH=./stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar
target_file=./data/test

out_file=./output/result.hypo

# Tokenize hypothesis and target files.
cat ${out_file} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${out_file}.tokenized

cat ${target_file}.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${target_file}.hypo.target

files2rouge ${out_file}.tokenized ${target_file}.hypo.target