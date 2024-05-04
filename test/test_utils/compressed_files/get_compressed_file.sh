gzip -c test > test.gz
gzip -c test > test.bgz
bzip2 -k test

tar -cvf test.tar test
tar -cvzf test.tar.gz test
tar -cvjf test.tar.bz2 test
zip test.zip test
