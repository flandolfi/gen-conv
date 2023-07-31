import logging

import fire

from .train import train, test

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    level=logging.INFO)


if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'test': test,
    })
