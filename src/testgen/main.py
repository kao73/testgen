import click

from testgen.di import DIContainer
from testgen.graph import MainGraph


@click.command()
def main():
    di = DIContainer()
    di.wire(packages=[
        'testgen',
        'testgen.graph',
        'testgen.tools',
    ])
    graph = MainGraph()
    response = graph.run({
        'source_folder': 'src',
        'target_folder': 'test'
    })
    print(response)


if __name__ == '__main__':
    main()
