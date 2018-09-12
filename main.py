from env import Game2048Env


def main():
    # arguments
    seed = 10

    env = Game2048Env(4)
    env.reset()
    env.render()
    board, reward, done, info = env.step(0)
    env.render()


if __name__ == "__main__":
    main()
