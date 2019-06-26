git checkout examples/example_machine_empty.json

rm -r outputs/machine 2> /dev/null

python3 RapidMain.py --flow TRAIN_ENV --path2machine ./examples/example_machine_empty.json --envdata ./testData/mmodelfile.csv --dir outputs/machine -d --test
