import torch

def eval(bets, trials):
    reward = 0

    dice = torch.randint(6, (trials, 3))

    for i in range(len(bets)):
        # Determine how many dice match per trial
        matches = (dice == i).int().sum(dim=1)

        # If matches, reward is the bet times the number of matches
        # If no matches, reward is the negative of the bet
        # Sum accross all trials
        reward += torch.where(matches > 0, bets[i] * matches, -bets[i]).sum()

    return reward / trials


print(eval([1,1,0,0,0,0], 10_000_000))

