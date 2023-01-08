from drone_path_planning.scenarios import Scenario


def train(scenario: Scenario):
    trainer = scenario.create_trainer()
    trainer.initialize()
    trainer.train()
