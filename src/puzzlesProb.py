import os
import chess
try:
    import chess.engine
except ImportError as e:
    print("Error: chess.engine module not found. Please install python-chess with: pip install python-chess>=0.30.0")
    raise e

import numpy as np
from jax import random as jrandom
import orbax.checkpoint as ocp
from src import transformer
from src import tokenizer, utils, training_utils
from src.engines import engine, neural_engines
import time
import csv

def create_predictor(model_size="270M", policy='action_value'):
    """Creates the transformer predictor with specified parameters based on model size."""
    num_return_buckets = 128

    # Configure model architecture based on size
    if model_size == "270M":
        num_layers = 16
        embedding_dim = 1024
        num_heads = 8
    elif model_size == "136M":
        num_layers = 8
        embedding_dim = 1024
        num_heads = 8
    elif model_size == "9M":
        num_layers = 8
        embedding_dim = 256
        num_heads = 8
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Configure output size based on policy
    if policy == 'action_value':
        output_size = num_return_buckets
    elif policy == 'state_value':
        output_size = num_return_buckets
    elif policy == 'behavioral_cloning':
        output_size = utils.NUM_ACTIONS
    else:
        raise ValueError(f"Unknown policy: {policy}")

    config = transformer.TransformerConfig(
        seed=1,  # Add this
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    return transformer.build_transformer_predictor(config=config)


def load_model(model_size="270M", policy='action_value'):
    """Loads the model with specified size and policy."""
    predictor = create_predictor(model_size, policy)
    if predictor is None:
        raise ValueError("Failed to create predictor")

    # Initialize parameters with dummy data
    dummy_params = predictor.initial_params(
        rng=jrandom.PRNGKey(0),
        targets=np.zeros((1, 1), dtype=np.uint32),
    )

    # Construct the path to checkpoint directory
    checkpoint_dir = os.path.join(
        os.getcwd(),
        '..',
        'checkpoints',
        model_size,
        '6400000'
    )

    print(f"Looking for checkpoint at: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    try:
        # Create Orbax checkpointer
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

        # Construct restore args from dummy params
        restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_params)

        # Load parameters directly from params_ema directory
        params = checkpointer.restore(
            os.path.join(checkpoint_dir, 'params_ema'),
            restore_args=restore_args
        )

        print(f"Successfully loaded parameters")
        return predictor, params

    except Exception as e:
        print(f"Error loading parameters: {str(e)}")
        print(f"Directory contents:")
        for root, dirs, files in os.walk(checkpoint_dir):
            print(f"\nDirectory: {root}")
            print("Files:", files)
            print("Subdirectories:", dirs)
        print(f"\nUsing dummy parameters for testing purposes")
        return predictor, dummy_params


def evaluate_positions(fen_list, model_size="270M",
                       time_limit=5,
                       batch_size=32):
    """
    Evaluates chess positions using specified model size and policy.
    """
    policy = 'state_value'
    predictor, params = load_model(model_size, policy)

    # Create appropriate engine based on policy
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
    predict_fn = neural_engines.wrap_predict_fn(
        predictor=predictor,
        params=params,
        batch_size=batch_size
    )
    engine_instance = neural_engines.StateValueEngine(
        return_buckets_values=return_buckets_values,
        predict_fn=predict_fn
    )

    results = {}
    for fen in fen_list:
        try:
            board = chess.Board(fen)
            analysis_list = []
            t0 = time.time()
            while True:
                analysis_list.append(engine_instance.analyse(board)['current_log_probs'])
                if time.time() - t0 > time_limit:
                    break
            log_probs = np.array(analysis_list)
            current_probs = np.exp(log_probs)
            win_probs = np.inner(current_probs, return_buckets_values)
            results[fen] = [float(np.mean(win_probs)), current_probs.shape[0]]

        except Exception as e:
            print(f"Error evaluating position {fen}: {str(e)}")
            print(f"Stack trace:", e.__traceback__)  # Added for debugging
            results[fen] = None

    return results


def create_stockfish_engine(time_limit=0.05):
    """
    Creates and configures a Stockfish engine instance.
    Returns None if Stockfish is not available.
    """
    try:
        # Look for Stockfish in common locations
        stockfish_paths = [
            os.path.join(os.getcwd(), '../Stockfish/src/stockfish'),
            # Project default
            '/usr/local/bin/stockfish',  # Common Unix location
            'stockfish'  # System PATH
        ]

        stockfish_path = None
        for path in stockfish_paths:
            if os.path.exists(path):
                stockfish_path = path
                break

        if stockfish_path is None:
            print(
                "Warning: Stockfish executable not found. Please install Stockfish and ensure it's in your PATH.")
            return None

        # Create engine with specified time limit
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        return engine

    except Exception as e:
        print(f"Error creating Stockfish engine: {str(e)}")
        return None


def evaluate_with_stockfish(fen_list, time_limit=0.05):
    """
    Evaluates chess positions using Stockfish.

    Args:
        fen_list: List of FEN strings to evaluate
        time_limit: Time limit per move in seconds (default 0.05 as used in paper)

    Returns:
        Dictionary mapping FEN strings to evaluation results
    """
    stockfish = create_stockfish_engine(time_limit)
    if stockfish is None:
        print("Skipping Stockfish evaluation due to missing engine")
        return {fen: None for fen in fen_list}

    results = {}
    try:
        for fen in fen_list:
            try:
                board = chess.Board(fen)
                # Analyze each legal move individually
                move_scores = []
                for move in board.legal_moves:
                    board.push(move)
                    result = stockfish.analyse(board, limit=chess.engine.Limit(
                        time=time_limit))
                    score = result["score"].relative
                    # Convert score to centipawns
                    if score.is_mate():
                        cp_score = 100 * score.mate()  # Arbitrary large value for mate
                    else:
                        cp_score = score.score()
                    win_prob = utils.centipawns_to_win_probability(cp_score)
                    move_scores.append((move.uci(), win_prob))
                    board.pop()

                move_scores.sort(key=lambda x: x[1], reverse=True)
                results[fen] = {
                    'win_probability': float(move_scores[0][1]),
                    'best_move': move_scores[0][0],
                    'all_moves': move_scores
                }

            except Exception as e:
                print(
                    f"Error evaluating position with Stockfish {fen}: {str(e)}")
                results[fen] = None

    finally:
        # Make sure to properly close the Stockfish engine
        if stockfish:
            stockfish.close()

    return results


def extract_columns(csv_file):
    column1 = []  # List to store column 1 (strings)
    column2 = []  # List to store column 2 (floats)

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header (if present)
        for row in reader:
            if len(row) >= 2:  # Ensure at least two columns
                column1.append(row[0])  # Add first column as string
                column2.append(float(row[1]))  # Add second column as float
    return column1, column2


def average_loss(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length.")

    total_loss = sum(abs(a - b) for a, b in zip(list1, list2))
    return total_loss / len(list1)


def main():
    test_path = "/Users/datkieu/PycharmProjects/searchless_chess/data/output.csv"
    test_positions, target = extract_columns(test_path)

    print("\nEvaluating with Transformer (136M model) under:")
    transformer_results = evaluate_positions(test_positions[:10], model_size="136M", time_limit=5)
    print(f"Average loss:{average_loss(transformer_results.values()[:][0], target[:10])}")


if __name__ == "__main__":
    main()
