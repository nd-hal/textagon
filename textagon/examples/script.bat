mprof run --multiprocess -C --output logs/dvd_FeatureConstruction_mprof.log process-text.py upload/dvd.txt dvd 0 0 4 16 upload/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 1 1 1 upload/exclusions.txt feature > logs/dvd_FeatureConstruction.log

mprof run --multiprocess -C --output logs/dvd_2gram_mprof.log process-text.py upload/dvd.txt dvd 0 0 2 16 upload/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 1 1 1 upload/exclusions.txt matrix > logs/dvd_2gram.log
