from TicTacToe import train
from ResumeTraining import train_resume
import torch

def main() -> None:
    # first instance training
    final_model, target, optimizer, buffer  = train()

    # train more deeply
    final_model = train_resume(final_model, target, optimizer, buffer, start_episode=2000000, num_episodes=20000)

    # Save the model
    torch.save(final_model.state_dict(), 'model_name.pth') # choose here model name

if __name__ == "__main__":
    main()