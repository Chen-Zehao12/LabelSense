#!/bin/bash

function main() {
    if [ $# -lt 1 -o "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Usage: bash $(basename $0) dataset_name"
        echo "Example: bash $(basename $0) 20Newsgroups"
        echo "Example: bash $(basename $0) all"
    fi

    if [ "$1" = "20Newsgroups" -o "$1" = "all" ]; then
        echo "Downloading 20Newsgroups"
        mkdir -p '20Newsgroups/origin'
        wget -O '20Newsgroups/origin/train.jsonl' 'https://huggingface.co/datasets/SetFit/20_newsgroups/resolve/main/train.jsonl'
        wget -O '20Newsgroups/origin/test.jsonl' 'https://huggingface.co/datasets/SetFit/20_newsgroups/resolve/main/test.jsonl'
    fi

    if [ "$1" = "Enron" -o "$1" = "all" ]; then
        echo "Downloading Enron"
        mkdir 'Enron'
        wget 'https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz'
        tar -xzf 'enron_mail_20150507.tar.gz'
        mv 'maildir' 'Enron/origin'
        rm 'enron_mail_20150507.tar.gz'
    fi

    if [ "$1" = "WOS" -o "$1" = "all" ]; then
        echo "Downloading WOS"
        mkdir 'WOS'
        wget 'https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9rw3vkcfy4-6.zip'
        unzip '9rw3vkcfy4-6.zip'
        unzip -d 'WOS/origin' 'WebOfScience.zip'
        rm '9rw3vkcfy4-6.zip' 'ReadMe.txt' 'WebOfScience.zip'
    fi

    if [ "$1" = "BGC" -o "$1" = "all" ]; then
        echo "Downloading BGC"
        mkdir 'BGC'
        wget 'https://fiona.uni-hamburg.de/ca89b3cf/blurbgenrecollectionen.zip'
        unzip -d 'BGC/origin' 'blurbgenrecollectionen.zip'
        rm 'blurbgenrecollectionen.zip'
    fi

    if [ "$1" = "RCV1-v2" -o "$1" = "all" ]; then
        echo "Downloading RCV1-v2"
        mkdir -p 'RCV1-v2/origin'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz'
        wget 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a16-rbb-topic/topics.rbb'
        gunzip 'rcv1-v2.topics.qrels.gz'
        gunzip 'lyrl2004_tokens_train.dat.gz'
        gunzip 'lyrl2004_tokens_test_pt0.dat.gz'
        gunzip 'lyrl2004_tokens_test_pt1.dat.gz'
        gunzip 'lyrl2004_tokens_test_pt2.dat.gz'
        gunzip 'lyrl2004_tokens_test_pt3.dat.gz'
        mv 'rcv1-v2.topics.qrels' 'RCV1-v2/origin'
        mv 'lyrl2004_tokens_train.dat' 'RCV1-v2/origin'
        mv 'lyrl2004_tokens_test_pt0.dat' 'RCV1-v2/origin'
        mv 'lyrl2004_tokens_test_pt1.dat' 'RCV1-v2/origin'
        mv 'lyrl2004_tokens_test_pt2.dat' 'RCV1-v2/origin'
        mv 'lyrl2004_tokens_test_pt3.dat' 'RCV1-v2/origin'
        mv 'topics.rbb' 'RCV1-v2/origin'
    fi

    if [ "$1" = "AAPD" -o "$1" = "all" ]; then
        echo "Downloading AAPD"
        mkdir -p 'AAPD/origin'
        wget -O 'AAPD.zip' 'https://git.uwaterloo.ca/jimmylin/Castor-data/-/archive/master/Castor-data-master.zip?path=datasets/AAPD/data'
        unzip 'AAPD.zip'
        rm 'AAPD.zip'
        mv Castor-data-master-datasets-AAPD-data/datasets/AAPD/data/* AAPD/origin/
        rm -rf 'Castor-data-master-datasets-AAPD-data'
    fi

    if [ "$1" = "RMSC" -o "$1" = "all" ]; then
        echo "Downloading RMSC"
        mkdir -p 'RMSC/origin'
        wget 'https://github.com/lancopku/RMSC/raw/master/data.zip'
        unzip 'AAPD.zip'
        rm 'data.zip'
        mv data/small/* RMSC/origin/
        rm -rf data
    fi

    if [ "$1" = "Reuters-21578" -o "$1" = "all" ]; then
        echo "Downloading Reuters-21578"
        mkdir -p 'Reuters-21578/origin'
        wget https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
        tar -xzf reuters21578.tar.gz -C Reuters-21578/origin
        rm reuters21578.tar.gz
    fi

    if [ "$1" = "freecode" -o "$1" = "all" ]; then
        echo "Please download freecode from kaggle: https://www.kaggle.com/datasets/navidkhezrian/freecode"
        echo "Unzip and move freecode_data.csv into freecode/origin"
        mkdir -p 'freecode/origin'
    fi

    if [ "$1" = "EUR-Lex" -o "$1" = "all" ]; then
        echo "Downloading EUR-Lex"
        mkdir -p 'EUR-Lex/origin'
        wget -O EUR-Lex/origin/eurovoc_concepts.jsonl https://archive.org/download/EURLEX57K/eurovoc_concepts.jsonl
        wget http://archive.org/download/EURLEX57K/dataset.zip
        unzip -d EUR-Lex/origin dataset.zip
        rm dataset.zip
    fi
}

main "$@"
