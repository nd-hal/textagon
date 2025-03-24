#!/bin/bash 

#$ -M rqin@nd.edu
#$ -m abe          
#$ -pe smp 2         # Specify parallel environment and legal core size
#$ -q long@@coba
#$ -N witnessconfidence      # Specify job name

conda activate mendoza


# python process-text.py upload/bs/Anxiety/raw.txt lexicon_v5/bs/Anxiety/Anxiety 0 0 1 4 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/AskRating/raw.txt lexicon_v5/bs/AskRating/AskRating 0 0 1 4 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/disastertweets_full/raw.txt lexicon_v5/bs/disastertweets_full/disastertweets_full 0 0 1 4 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/distress/raw.txt lexicon_v5/bs/distress/distress 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/drug/raw.txt lexicon_v5/bs/drug/drug 0 0 1 4 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/empathy/raw.txt lexicon_v5/bs/empathy/empathy 0 0 1 4 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/jigsaw_20k_full/raw.txt lexicon_v5/bs/jigsaw_20k_full/jigsaw_20k_full 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/Numeracy/raw.txt lexicon_v5/bs/Numeracy/Numeracy 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/quora_20k_full/raw.txt lexicon_v5/bs/quora_20k_full/quora_20k_full 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/SubjectiveLit/raw.txt lexicon_v5/bs/SubjectiveLit/SubjectiveLit 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/TrustPhys/raw.txt lexicon_v5/bs/TrustPhys/TrustPhys 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/Tweets_ADRsASU/raw.txt lexicon_v5/bs/Tweets_ADRsASU/Tweets_ADRsASU 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/witnessaccuracy/raw.txt lexicon_v5/bs/witnessaccuracy/witnessaccuracy 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full

# python process-text.py upload/bs/witnessconfidence/raw.txt lexicon_v5/bs/witnessconfidence/witnessconfidence 0 0 1 2 external/lexicons/Lexicons_v5.zip 1 5 bB 0 1 0 3 1 1 0 1 1 upload/exclusions.txt full