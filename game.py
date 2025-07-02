import random
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION - Easily customizable parameters
# =============================================================================

@dataclass
class GameConfig:
    """Main configuration for the game - modify these values easily"""
    # Player stats
    MAX_HEALTH: int = 100
    
    # Learning parameters
    LEARNING_RATE: float = 0.1
    DISCOUNT_FACTOR: float = 0.95
    EPSILON_START: float = 0.9  # Exploration rate
    EPSILON_DECAY: float = 0.995
    EPSILON_MIN: float = 0.01
    
    # Training
    BOT_TRAINING_EPISODES: int = 1000
    
    # File paths
    AI_SAVE_FILE: str = "ai_brain.json"

@dataclass
class MoveConfig:
    """Configuration for moves - easily add/remove/modify moves here"""
    name: str
    damage: int
    accuracy: float  # 0.0 to 1.0
    heal_amount: int = 0  # 0 for attack moves
    move_type: str = "attack"  # "attack" or "heal"

# Define all available moves here - easily customizable
AVAILABLE_MOVES = [
    MoveConfig("Heavy Strike", damage=40, accuracy=0.8, move_type="attack"),
    MoveConfig("Quick Jab", damage=25, accuracy=0.95, move_type="attack"),
    MoveConfig("Heal", damage=0, accuracy=1.0, heal_amount=20, move_type="heal"),
    # Add more moves here easily:
    # MoveConfig("Lightning Bolt", damage=40, accuracy=0.5, move_type="attack"),
    # MoveConfig("Shield Bash", damage=20, accuracy=0.8, move_type="attack"),
    # MoveConfig("Major Heal", damage=0, accuracy=1.0, heal_amount=40, move_type="heal"),
]

# =============================================================================
# CORE GAME CLASSES
# =============================================================================

class Player:
    """Represents a player in the battle"""
    
    def __init__(self, name: str, config: GameConfig):
        self.name = name
        self.max_health = config.MAX_HEALTH
        self.health = self.max_health
        self.config = config
    
    def take_damage(self, damage: int) -> int:
        """Apply damage and return actual damage taken"""
        actual_damage = min(damage, self.health)
        self.health -= actual_damage
        return actual_damage
    
    def heal(self, amount: int) -> int:
        """Heal and return actual amount healed"""
        actual_heal = min(amount, self.max_health - self.health)
        self.health += actual_heal
        return actual_heal
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def reset_health(self):
        self.health = self.max_health
    
    def get_health_percentage(self) -> float:
        return self.health / self.max_health


class Move:
    """Represents a battle move"""
    
    def __init__(self, config: MoveConfig):
        self.config = config
    
    def execute(self, user: Player, target: Player) -> Tuple[bool, str]:
        """Execute the move and return (success, message)"""
        # Check if move hits
        if random.random() > self.config.accuracy:
            return False, f"{user.name} used {self.config.name} but missed!"
        
        if self.config.move_type == "attack":
            damage_dealt = target.take_damage(self.config.damage)
            return True, f"{user.name} used {self.config.name} and dealt {damage_dealt} damage!"
        
        elif self.config.move_type == "heal":
            heal_amount = user.heal(self.config.heal_amount)
            return True, f"{user.name} used {self.config.name} and healed {heal_amount} HP!"
        
        return False, "Unknown move type!"

# =============================================================================
# REINFORCEMENT LEARNING AI
# =============================================================================

class QLearningAI:
    """Q-Learning based AI that learns from battles"""
    
    def __init__(self, config: GameConfig, moves: List[MoveConfig]):
        self.config = config
        self.moves = moves
        self.q_table = {}  # State -> Action values
        self.epsilon = config.EPSILON_START
        self.learning_rate = config.LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        
        # Experience for learning
        self.last_state = None
        self.last_action = None
    
    def get_state(self, my_health: int, enemy_health: int) -> str:
        """Convert game state to string key for Q-table"""
        # Discretize health into ranges for manageable state space
        my_health_range = min(my_health // 20, 5)  # 0-5 ranges
        enemy_health_range = min(enemy_health // 20, 5)
        return f"{my_health_range}_{enemy_health_range}"
    
    def get_q_values(self, state: str) -> List[float]:
        """Get Q-values for all actions in given state"""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(self.moves)
        return self.q_table[state]
    
    def choose_action(self, my_health: int, enemy_health: int, training: bool = False) -> int:
        """Choose action using epsilon-greedy strategy"""
        state = self.get_state(my_health, enemy_health)
        q_values = self.get_q_values(state)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            action = random.randint(0, len(self.moves) - 1)
        else:
            action = np.argmax(q_values)
        
        self.last_state = state
        self.last_action = action
        return action
    
    def learn(self, reward: float, my_health: int, enemy_health: int, game_over: bool):
        """Update Q-values based on experience"""
        if self.last_state is None or self.last_action is None:
            return
        
        current_state = self.get_state(my_health, enemy_health)
        current_q_values = self.get_q_values(current_state)
        
        # Q-learning update
        if game_over:
            target = reward
        else:
            target = reward + self.discount_factor * max(current_q_values)
        
        old_q_values = self.get_q_values(self.last_state)
        old_q_values[self.last_action] += self.learning_rate * (target - old_q_values[self.last_action])
        
        # Decay epsilon
        if self.epsilon > self.config.EPSILON_MIN:
            self.epsilon *= self.config.EPSILON_DECAY
    
    def reset_episode(self):
        """Reset episode-specific data"""
        self.last_state = None
        self.last_action = None
    
    def save_brain(self, filename: str):
        """Save Q-table to file"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_brain(self, filename: str) -> bool:
        """Load Q-table from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.q_table = data.get('q_table', {})
                self.epsilon = data.get('epsilon', self.config.EPSILON_START)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

# =============================================================================
# GAME LOGIC
# =============================================================================

class BattleGame:
    """Main game controller"""
    
    def __init__(self, config: GameConfig = None):
        self.config = config or GameConfig()
        self.moves = [Move(move_config) for move_config in AVAILABLE_MOVES]
        self.move_configs = AVAILABLE_MOVES
        
        # Initialize AI
        self.ai = QLearningAI(self.config, self.move_configs)
        self.ai.load_brain(self.config.AI_SAVE_FILE)
        
        # Game stats
        self.player_wins = 0
        self.ai_wins = 0
    
    def display_moves(self):
        """Display available moves"""
        print("\nAvailable Moves:")
        for i, move_config in enumerate(self.move_configs):
            if move_config.move_type == "attack":
                print(f"{i+1}. {move_config.name} - Damage: {move_config.damage}, Accuracy: {move_config.accuracy:.0%}")
            else:
                print(f"{i+1}. {move_config.name} - Heal: {move_config.heal_amount}, Accuracy: {move_config.accuracy:.0%}")
    
    def get_player_action(self) -> int:
        """Get player's move choice"""
        while True:
            try:
                choice = int(input(f"\nChoose your move (1-{len(self.moves)}): ")) - 1
                if 0 <= choice < len(self.moves):
                    return choice
                else:
                    print(f"Please enter a number between 1 and {len(self.moves)}")
            except ValueError:
                print("Please enter a valid number")
    
    def display_status(self, player: Player, ai_player: Player):
        """Display current battle status"""
        print(f"\n{'='*50}")
        print(f"{player.name}: {player.health}/{player.max_health} HP")
        print(f"AI: {ai_player.health}/{ai_player.max_health} HP")
        print(f"{'='*50}")
    
    def calculate_reward(self, ai_health_before: int, ai_health_after: int, 
                        player_health_before: int, player_health_after: int,
                        ai_won: bool, player_won: bool) -> float:
        """Calculate reward for AI learning"""
        reward = 0.0
        
        # Health change rewards
        health_lost = ai_health_before - ai_health_after
        health_gained = ai_health_after - ai_health_before
        enemy_health_lost = player_health_before - player_health_after
        
        reward -= health_lost * 0.1  # Penalty for taking damage
        reward += health_gained * 0.1  # Reward for healing
        reward += enemy_health_lost * 0.1  # Reward for dealing damage
        
        # Game outcome rewards
        if ai_won:
            reward += 10.0
        elif player_won:
            reward -= 10.0
        
        return reward
    
    def play_battle(self, training_mode: bool = False) -> str:
        """Play a single battle. Returns winner."""
        player = Player("Player", self.config)
        ai_player = Player("AI", self.config)
        
        self.ai.reset_episode()
        
        if not training_mode:
            print(f"\n🎮 BATTLE START! 🎮")
            print(f"You vs AI (Wins: {self.player_wins} - {self.ai_wins})")
        
        turn = 0
        while player.is_alive() and ai_player.is_alive():
            turn += 1
            
            if not training_mode:
                self.display_status(player, ai_player)
                self.display_moves()
            
            # Store health for reward calculation
            ai_health_before = ai_player.health
            player_health_before = player.health
            
            # Player turn
            if not training_mode:
                player_action = self.get_player_action()
            else:
                # In training mode, player is also AI (random for simplicity)
                player_action = random.randint(0, len(self.moves) - 1)
            
            success, message = self.moves[player_action].execute(player, ai_player)
            if not training_mode:
                print(f"\n{message}")
            
            if not ai_player.is_alive():
                break
            
            # AI turn
            ai_action = self.ai.choose_action(ai_player.health, player.health, training_mode)
            success, message = self.moves[ai_action].execute(ai_player, player)
            if not training_mode:
                print(f"{message}")
            
            # AI learning
            ai_health_after = ai_player.health
            player_health_after = player.health
            
            reward = self.calculate_reward(
                ai_health_before, ai_health_after,
                player_health_before, player_health_after,
                not player.is_alive(), not ai_player.is_alive()
            )
            
            self.ai.learn(reward, ai_player.health, player.health, 
                         not player.is_alive() or not ai_player.is_alive())
        
        # Determine winner
        if player.is_alive():
            winner = "Player"
            self.player_wins += 1
        else:
            winner = "AI"
            self.ai_wins += 1
        
        if not training_mode:
            print(f"\n🏆 {winner} wins! 🏆")
            self.display_status(player, ai_player)
        
        return winner
    
    def train_ai_with_bots(self, episodes: int = None):
        """Train AI by playing against random bots"""
        episodes = episodes or self.config.BOT_TRAINING_EPISODES
        print(f"\n🤖 Training AI with {episodes} bot battles...")
        
        wins = 0
        for episode in range(episodes):
            if episode % 100 == 0 and episode > 0:
                print(f"Training progress: {episode}/{episodes} (AI win rate: {wins/episode:.2%})")
            
            winner = self.play_battle(training_mode=True)
            if winner == "AI":
                wins += 1
        
        print(f"✅ Training complete! AI won {wins}/{episodes} battles ({wins/episodes:.2%})")
        self.ai.save_brain(self.config.AI_SAVE_FILE)
        print(f"💾 AI brain saved to {self.config.AI_SAVE_FILE}")
    
    def run_game(self):
        """Main game loop"""
        print("🎮 Welcome to RL Battle Game! 🎮")
        print("The AI learns from every battle and gets stronger!")
        
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Play Battle")
            print("2. Train AI with Bots")
            print("3. View Stats")
            print("4. Reset AI Brain")
            print("5. Quit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self.play_battle()
                self.ai.save_brain(self.config.AI_SAVE_FILE)
                
            elif choice == "2":
                episodes = input(f"Enter number of training episodes (default {self.config.BOT_TRAINING_EPISODES}): ").strip()
                try:
                    episodes = int(episodes) if episodes else self.config.BOT_TRAINING_EPISODES
                    self.train_ai_with_bots(episodes)
                except ValueError:
                    print("Invalid number, using default.")
                    self.train_ai_with_bots()
                    
            elif choice == "3":
                print(f"\n📊 GAME STATS 📊")
                print(f"Player Wins: {self.player_wins}")
                print(f"AI Wins: {self.ai_wins}")
                total_games = self.player_wins + self.ai_wins
                if total_games > 0:
                    print(f"Player Win Rate: {self.player_wins/total_games:.2%}")
                    print(f"AI Win Rate: {self.ai_wins/total_games:.2%}")
                print(f"AI Learning Progress: {len(self.ai.q_table)} states learned")
                print(f"AI Exploration Rate: {self.ai.epsilon:.3f}")
                
            elif choice == "4":
                confirm = input("Are you sure you want to reset the AI brain? (y/N): ").strip().lower()
                if confirm == 'y':
                    self.ai = QLearningAI(self.config, self.move_configs)
                    if os.path.exists(self.config.AI_SAVE_FILE):
                        os.remove(self.config.AI_SAVE_FILE)
                    print("🧠 AI brain reset!")
                    
            elif choice == "5":
                print("Thanks for playing! 👋")
                break
                
            else:
                print("Invalid choice. Please try again.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the game"""
    # You can customize the game config here
    config = GameConfig()
    
    # Example of customizing parameters:
    # config.MAX_HEALTH = 150
    # config.LEARNING_RATE = 0.2
    # config.BOT_TRAINING_EPISODES = 2000
    
    game = BattleGame(config)
    game.run_game()

if __name__ == "__main__":
    main()