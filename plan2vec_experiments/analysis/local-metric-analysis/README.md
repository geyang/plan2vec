# How to Run This

In order for the Street Learn code to run on 
learnfair, you need to have the dataset on your
devfair.

1. inside your devfair

    ```shell script
    source ~/.profile
    module load cuda/10.0
    module load cudnn/v7.4-cuda.10.0
    module load anaconda3
    source activate plan2vec

    pip install --upgrade waterbear params-proto
    pip install --upgrade ml-logger
    ```

2. Now on your local computer

    ```shell script
    cd ~
    mkdir fair
    cd fair
    git clone https://github.com/episodeyang/plan2vec
    git clone https://github.com/episodeyang/ge_world
    git clone https://github.com/episodeyang/streetlearn

    cd streetlearn
    git 
    ```

3. You need a copy of the `jaynes.yml` config file
4. run `all_local_metric.py` file

To make sure things are running: use logger to check the results