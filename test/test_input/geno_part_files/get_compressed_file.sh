gzip -c geno_part > geno_part.gz
gzip -c geno_part > geno_part.bgz
bzip2 -k geno_part

tar -cvf geno_part.tar geno_part
tar -cvzf geno_part.tar.gz geno_part
tar -cvjf geno_part.tar.bz2 geno_part
zip geno_part.zip geno_part
