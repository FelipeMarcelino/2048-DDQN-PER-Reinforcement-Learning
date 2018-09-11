
from game import Game2048


def main():
    # arguments
    seed = 10

    game = Game2048(4)

    print(game.check_available_moves())


if __name__ == "__main__":
    main()
