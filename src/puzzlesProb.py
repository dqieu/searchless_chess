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

from searchless_chess.src import tokenizer
from searchless_chess.src import transformer
from searchless_chess.src import utils
from searchless_chess.src import training_utils
from searchless_chess.src.engines import neural_engines
from searchless_chess.src.engines import engine


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

def evaluate_positions(fen_list, model_size="270M", policy='action_value',
                       batch_size=32):
    """
    Evaluates chess positions using specified model size and policy.
    """
    predictor, params = load_model(model_size, policy)

    # Create appropriate engine based on policy
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(128)
    predict_fn = neural_engines.wrap_predict_fn(
        predictor=predictor,
        params=params,
        batch_size=batch_size
    )

    if policy == 'action_value':
        engine_instance = neural_engines.ActionValueEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn
        )
    elif policy == 'state_value':
        engine_instance = neural_engines.StateValueEngine(
            return_buckets_values=return_buckets_values,
            predict_fn=predict_fn
        )
    else:  # behavioral_cloning
        engine_instance = neural_engines.BCEngine(predict_fn=predict_fn)

    results = {}
    for fen in fen_list:
        try:
            board = chess.Board(fen)
            analysis = engine_instance.analyse(board)

            if policy == 'action_value':
                buckets_log_probs = analysis['log_probs']
                buckets_probs = np.exp(buckets_log_probs)
                win_probs = np.inner(buckets_probs, return_buckets_values)

                sorted_legal_moves = engine.get_ordered_legal_moves(board)

                move_probs = [(move.uci(), prob) for move, prob in
                              zip(sorted_legal_moves, win_probs)]
                move_probs.sort(key=lambda x: x[1], reverse=True)

                results[fen] = {
                    'win_probability': float(move_probs[0][1]),
                    'best_move': move_probs[0][0],
                    'all_moves': move_probs
                }

            elif policy == 'state_value':
                current_log_probs = analysis['current_log_probs']
                current_probs = np.exp(current_log_probs)
                win_prob = np.inner(current_probs, return_buckets_values)

                results[fen] = float(win_prob)

            else:  # behavioral_cloning
                action_log_probs = analysis['log_probs']
                action_probs = np.exp(action_log_probs)

                sorted_legal_moves = engine.get_ordered_legal_moves(board)
                move_probs = [(move.uci(), prob) for move, prob in
                              zip(sorted_legal_moves, action_probs)]
                move_probs.sort(key=lambda x: x[1], reverse=True)

                results[fen] = {
                    'best_move': move_probs[0][0],
                    'move_probabilities': move_probs
                }

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


# [Previous code for transformer evaluation remains the same]

def main():
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        # Starting position
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        # Common position
    ]

    # First try with dummy parameters to ensure rest of pipeline works
    try:
        print("\nEvaluating with Transformer (270M model):")
        transformer_results = evaluate_positions(test_positions,
                                                 model_size="270M",
                                                 policy='action_value')

        print("\nEvaluating with Stockfish (baseline):")
        stockfish_results = evaluate_with_stockfish(test_positions)

        # Print comparative results
        for fen in test_positions:
            print(f"\nPosition: {fen}")
            print("----------------------------------------")

            # Transformer results
            t_result = transformer_results.get(fen)
            if t_result:
                print("Transformer evaluation:")
                print(f"Win probability: {t_result['win_probability']:.1%}")
                print(f"Best move: {t_result['best_move']}")
                print("Top 3 moves with probabilities:")
                for move, prob in t_result['all_moves'][:3]:
                    print(f"{move}: {prob:.1%}")
            else:
                print("Transformer evaluation failed")

            print()

            # Stockfish results
            s_result = stockfish_results.get(fen)
            if s_result:
                print("Stockfish evaluation (baseline):")
                print(f"Win probability: {s_result['win_probability']:.1%}")
                print(f"Best move: {s_result['best_move']}")
                print("Top 3 moves with probabilities:")
                for move, prob in s_result['all_moves'][:3]:
                    print(f"{move}: {prob:.1%}")
            else:
                print("Stockfish evaluation failed")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
