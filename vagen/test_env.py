from vagen.env import get_env, get_service

if __name__ == "__main__":
    env = get_env("alfred")
    service = get_service("alfred")
    print(env)
    print(service)