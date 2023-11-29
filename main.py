from drawer import run_app
from model_cnn import initialise_model

if __name__ == "__main__":
    choice = input("Train new model (1)\nUse existing model (2)\n-> ")
    if choice == "1":
        initialise_model()
    elif choice == "2":
        run_app()
    else:
        print("Invalid choice")
