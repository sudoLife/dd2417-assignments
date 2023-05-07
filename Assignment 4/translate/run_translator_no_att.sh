CUBLAS_WORKSPACE_CONFIG=:4096:8 python translate.py -tr eng-swe-train.txt -de eng-swe-dev.txt -te eng-swe-test.txt -hs 100 -bs 40 -e 41 -lr 0.001 -ef glove.6B.50d.txt -b -g -s -et
