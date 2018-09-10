
from game import Game2048


def main():
    # arguments
    seed = 10

    game = Game2048(4, seed)
    print(game.get_board())


if __name__ == "__main__":
    main()
