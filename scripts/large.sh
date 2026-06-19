python pretrain_hyper.py ShiFT ecg2 -p configs/ecg2config.yml -s 1 --evaluate supervised
python pretrain_hyper.py ShiFT harth -p configs/harthconfig.yml -s 1 --evaluate supervised
python pretrain_hyper.py ShiFT pamap2 -p configs/pamap2config.yml -s 1 --evaluate supervised
python pretrain_hyper.py ShiFT skoda -p configs/skodaconfig.yml -s 1 --evaluate supervised
python pretrain_hyper.py ShiFT sleepm -p configs/sleepmconfig.yml -s 1 --evaluate supervised
python pretrain_hyper.py ShiFT wisdm2 -p configs/wisdm2config.yml -s 1 --evaluate supervised


python launch_all.py
python evaluate_all.py
python evaluate_clustering.py