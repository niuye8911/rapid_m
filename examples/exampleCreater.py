import json


def createExampleApp():
    example_app = dict()
    example_app['name'] = "exampleApp"
    example_app['TRAINED'] = False
    example_app['CLUSTERED'] = False
    example_app['machine_id'] = 0
    with open('../examples/example_app_empty.json', 'w') as example_app_file:
        json.dump(example_app, example_app_file, indent=2)


def createExampleAppAfterTrained():
    with open('../examples/example_app_empty.json', 'r') as example_app_file:
        example_app = json.load(example_app_file)
        example_app['params'] = {"CPU": 0.1, "IO": 0.2}
        example_app['TRAINED'] = True
        example_app['CLUSTERED'] = True
        example_app['accuracy'] = 0.80
        example_app['num_of_cluster'] = 3
        example_app['cluster_info'] = {"first": list("config1", "config2"), "second": list("config3", "config4")}
    with open('../examples/example_app_trained.json', 'w') as example_app_file:
        json.dump(example_app, example_app_file, indent=2)


createExampleApp()
createExampleAppAfterTrained()
