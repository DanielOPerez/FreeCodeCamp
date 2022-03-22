previous_steps = {}
best_move={"P": "S", "R": "P", "S": "R"}

def player(prev_play, opponent_history=[]):
    if prev_play != "":
        opponent_history.append(prev_play)
      
    #number of previous previous_steps to watch
    n = 6
     
    #initialice guess
    guess = "S"
  
    if len(opponent_history) >= n:
        pattern = "".join(opponent_history[-n:])
        
        if "".join(opponent_history[-(n + 1):]) in previous_steps.keys():
            previous_steps["".join(opponent_history[-(n + 1):])] += 1
        else:
            previous_steps["".join(opponent_history[-(n + 1):])] = 1

        next_movement = [pattern + "R", 
                         pattern + "P", 
                         pattern + "S"]
          
        for i in next_movement:
            if i not in previous_steps:
              previous_steps[i] = 0
        
        predict = max(next_movement, key=lambda x: previous_steps[x])
  
        guess = best_move[predict[-1]]
        
       
      
    return guess
