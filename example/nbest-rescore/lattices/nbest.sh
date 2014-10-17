ls *.lat.gz > latlist

lattice-tool \
-nbest-decode 5 \
-read-htk \
-htk-logbase 2.718 \
-htk-lmscale 14 \
-htk-wdpenalty 0.0 \
-in-lattice-list latlist \
-out-nbest-dir nbest
