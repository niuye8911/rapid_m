1) INIT app
python3 RapidMain.py --flow INIT --path2app examples/example_app_empty.json --apppfs testData/lighter/facedetect-sys.csv --appdata testData/lighter/facedetect-perf.csv --dir outputs/facedetect


2) INIT machine
python3 RapidMain.py --flow TRAIN_ENV --path2machine ./examples/example_machine_empty.json --envdata ./mmodelfile.csv --dir outputs/machine


3) validate M / P model
python3 validator.py --macsum ./examples/example_machine_empty.json --appsum ./outputs/ferret/profile.json --appsys ./testData/lighter/ferret-sys.csv --obs ./testData/lighter/ferret-mperf.csv
