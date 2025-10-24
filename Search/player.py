#!/usr/bin/env python3
import math
from time import time
from collections import namedtuple, OrderedDict

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

# Transposition table entry
TTEntry = namedtuple('TTEntry', 'value depth flag best_move')
EXACT, LOWER, UPPER = 0, 1, 2



class SearchTimeout(Exception):
    pass


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        while True:
            msg = self.receiver()
            if msg.get("game_over"):
                return


class PlayerControllerMinimax(PlayerController):
    """
    Minimax (alpha-beta) with iterative deepening, LRU transposition table,
    toroidal Manhattan distances and lightweight micro-optimizations.
    """

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

        self._safety_fraction = 0.80   # use 80% of the allowed time
        self._depth_cap = 8            # absolute maximum depth
        self._beam_k = 2               # at node, keep only K best moves for deeper iterations
        self._eval_top_fish = 6        # evaluate only the N most relevant fish

        # runtime fields
        self._start_time = 0.0
        self._deadline = 0.0

        self.tt = OrderedDict()      # transposition table (LRU - Least Recently Used) - it stores the alredy seen states
        self._tt_size_limit = 20000  # cap size
        self.eval_cache = {}         # cache for storing the values of the heuristic function, cleared every turn
        self.prev_pv = {}            # principal variation move per depth (ordering)
        self.node_count = 0          # diagnostics

    # -------------------- utilities --------------------
    def make_key_from_node(self, node):
        """
        Build a hash key from node.message in a deterministic way.
        The key is buil on hooks, fish positions, scores and our caught fish id.
        """
        m = getattr(node, "message", None)
        if m is None:
            return None
        try:
            hooks = m.get("hooks_positions", {})
            h0 = hooks.get(0, (0, 0))
            h1 = hooks.get(1, (0, 0))
            hp = (h0[0], h0[1], h1[0], h1[1])

            fishes = m.get("fishes_positions", {})
            fish_list = []
            for fid in sorted(fishes.keys()):
                x, y = fishes[fid]
                fish_list.append(x)
                fish_list.append(y)

            p_scores = m.get("player_scores", {})
            if isinstance(p_scores, dict):
                p0 = p_scores.get(0, 0)
                p1 = p_scores.get(1, 0)
            else:
                p0 = p_scores[0] if len(p_scores) > 0 else 0
                p1 = p_scores[1] if len(p_scores) > 1 else 0


            caught = m.get("caught_fish", {0: None, 1: None})
            my_caught = None
            if isinstance(caught, dict):
                my_caught = caught.get(0)
            else:
                try:
                    my_caught = caught[0]
                except Exception:
                    my_caught = None

            return (hp, tuple(fish_list), p0, p1, my_caught)
        except Exception:
            return None

    # -------------------- main loop --------------------
    def player_loop(self):

        # initial message received
        first_msg = self.receiver()

        while True:
            msg = self.receiver()
            if msg.get("game_over"):
                return

            self._start_time = time()
            budget = getattr(self.settings, "time_threshold", 0.075)
            self._deadline = self._start_time + self._safety_fraction * budget

            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            # if we already have a caught fish, we simply go "up"
            if msg["caught_fish"][0] is not None:
                best_move = "up"
            else:            
                best_move = self.search_with_iterative_deepening(node)

            self.sender({"action": best_move, "search_time": time() - self._start_time})

    
    # -------------------- search core --------------------
    def search_with_iterative_deepening(self, node):
        # firstly the cache is cleared
        self.eval_cache.clear()
        self.node_count = 0

        node_children = self.children_with_actions(node)
        if not node_children:
            return "stay"

        # initial ordering for enhancing pruning
        node_children_with_eval = []
        for act, ch in node_children:
            try:
                val = self.evaluate(ch)
            except Exception:
                val = 0.0
            node_children_with_eval.append((val, act, ch))

        node_children_with_eval.sort(key=lambda x: -x[0])
        node_children = [(a, c) for _, a, c in node_children_with_eval]

        best_action = node_children[0][0]   #default best action at the root
        bf = len(node_children)

        completed_depth = 0
        d = 1
        try:
            while d <= self._depth_cap:
                if time() >= self._deadline:
                    break

                # beam at root to control branching for deeper iterations
                if d >= 3 and bf > self._beam_k:
                    frontier = node_children[: self._beam_k]
                else:
                    frontier = node_children

                alpha, beta = -math.inf, math.inf
                current_best_val = -math.inf
                current_best_act = best_action

                for act, child in frontier:
                    if time() >= self._deadline:
                        raise SearchTimeout()
                    val = self.alphabeta(child, d - 1, alpha, beta, maximizing=False)
                    if val > current_best_val:
                        current_best_val, current_best_act = val, act
                    alpha = max(alpha, current_best_val)

                best_action = current_best_act
                self.prev_pv[d] = best_action
                completed_depth = d
                d += 1

        except SearchTimeout:
            pass

        return best_action




    def alphabeta(self, node, depth, alpha, beta, maximizing):
        # local bindings for increasing speed
        time_fn = time
        deadline = self._deadline
        eval_fn = self.evaluate
        children_fn = self.children_with_actions
        tt = self.tt

        # occasional quick time check
        if time_fn() >= deadline:
            raise SearchTimeout()

        # probe TT (LRU semantics)
        key = self.make_key_from_node(node)
        if key is not None:
            entry = tt.get(key)
            if entry is not None:
                # move to end (recent)
                try:
                    del tt[key]
                    tt[key] = entry
                except Exception:
                    pass
                if entry.depth >= depth:
                    if entry.flag == EXACT:
                        return entry.value
                    if entry.flag == LOWER and entry.value >= beta:
                        return entry.value
                    if entry.flag == UPPER and entry.value <= alpha:
                        return entry.value

        if depth == 0:
            return eval_fn(node)

        children = children_fn(node)
        if not children:
            return eval_fn(node)

        self.node_count += 1

        # ordering: prefer TT.best_move at front
        tt_best = None
        if key is not None:
            e = tt.get(key)
            if e is not None:
                tt_best = e.best_move

        scored = []
        for act, ch in children:
            score = 0
            if act == tt_best:
                score += 100000
            try:
                score += 0.1 * eval_fn(ch)
            except Exception:
                pass
            scored.append((score, act, ch))

        if maximizing:
            scored.sort(key=lambda x: -x[0])
        else:
            scored.sort(key=lambda x: x[0])

        orig_alpha = alpha
        orig_beta = beta
        best_move_for_node = None

        # reduce frequency of time() calls by checking every N iterations
        time_check_counter = 0
        if maximizing:
            value = -math.inf
            for _, act, ch in scored:
                time_check_counter += 1
                if (time_check_counter & 7) == 0 and time_fn() >= deadline:
                    raise SearchTimeout()

                v = self.alphabeta(ch, depth - 1, alpha, beta, maximizing=False)
                if v > value:
                    value = v
                    best_move_for_node = act
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for _, act, ch in scored:
                time_check_counter += 1
                if (time_check_counter & 7) == 0 and time_fn() >= deadline:
                    raise SearchTimeout()

                v = self.alphabeta(ch, depth - 1, alpha, beta, maximizing=True)
                if v < value:
                    value = v
                    best_move_for_node = act
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # store in TT (LRU insert)
        try:
            if key is not None:
                if value <= orig_alpha:
                    flag = UPPER
                elif value >= orig_beta:
                    flag = LOWER
                else:
                    flag = EXACT
                if key in tt:
                    del tt[key]
                tt[key] = TTEntry(value=value, depth=depth, flag=flag, best_move=best_move_for_node)
                if len(tt) > self._tt_size_limit:
                    tt.popitem(last=False)
        except Exception:
            pass

        return value

    def children_with_actions(self, node):
        kids = node.compute_and_get_children()
        out = []
        for item in kids:
            if isinstance(item, tuple) and len(item) == 2:
                act, ch = item
                act_str = ACTION_TO_STR[act] if isinstance(act, int) else str(act)
                out.append((act_str, ch))
            else:
                ch = item
                move = getattr(ch, "move", 0)
                act_str = ACTION_TO_STR[move] if isinstance(move, int) else str(move)
                out.append((act_str, ch))
        return out

    def evaluate(self, node):
        """
        Heuristic: 10*score_margin + positional advantage based on toroidal Manhattan distance.
        No carry bonus, no heavy simulations. Lightweight, cached by compact key.
        """
        # helper: toroidal Manhattan
        def toroidal_manhattan(a, b, w, h):
            dx = abs(a[0] - b[0])
            dx = min(dx, max(1, w) - dx)
            dy = abs(a[1] - b[1])
            return dx + dy

        key = self.make_key_from_node(node)
        if key is not None:
            v = self.eval_cache.get(key)
            if v is not None:
                return v

        s = getattr(node, 'state', None)
        try:
            if s is not None:
                p0, p1 = s.get_player_scores()
                hooks = s.get_hook_positions()
                fishes = s.get_fish_positions() or {}
                values = s.get_fish_scores() or {}
                msg = getattr(node, "message", {}) or {}
            else:
                m = getattr(node, 'message', None)
                if m is None:
                    return 0.0
                p0 = m['player_scores'][0]
                p1 = m['player_scores'][1]
                hooks = m['hooks_positions']
                fishes = m['fishes_positions']
                values = m['fish_scores']
                msg = m

            # infer board size from known positions
            max_x = 0
            max_y = 0
            try:
                for pos in hooks.values():
                    max_x = max(max_x, pos[0])
                    max_y = max(max_y, pos[1])
            except Exception:
                pass
            try:
                for pos in fishes.values():
                    max_x = max(max_x, pos[0])
                    max_y = max(max_y, pos[1])
            except Exception:
                pass
            board_w = max(20, max_x + 1)
            board_h = max(20, max_y + 1)

            # compute margin and positional bonus
            margin = p0 - p1
            positional = 0.0
            my = hooks[0]
            opp = hooks[1]

            rel = []
            for fid, pos in fishes.items():
                val = values.get(fid, 0)
                if val == 0:
                    continue
                d_me = toroidal_manhattan(my, pos, board_w, board_h)
                d_op = toroidal_manhattan(opp, pos, board_w, board_h)
                rel_score = val / (1.0 + min(d_me, d_op))
                rel.append((rel_score, fid, d_me, d_op, val))

            rel.sort(reverse=True)
            rel = rel[: self._eval_top_fish]

            for _, fid, d_me, d_op, val in rel:
                positional += (val / (1.0 + d_me)) - 0.8 * (val / (1.0 + d_op))
                if d_me + 2 <= d_op:
                    positional += 0.3 * val

            score = 10.0 * margin + positional
        except Exception:
            score = 0.0

        if key is not None:
            self.eval_cache[key] = score
        return score
