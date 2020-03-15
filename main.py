import argparse
import time
from collections import defaultdict
from heapq import heappop, heappush

import gym
import gym_minigrid
import numpy as np
from gym_minigrid.window import Window
from gym_minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper


def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

class Node(object):
    def __init__(self, name, node_type, i, j, adj=None):
        self.name = name # should be unique
        self.type = node_type
        self.i = i
        self.j = j
        self.adj = [] # adj list

    def __str__(self):
        return f'({self.name})'

    def __repr__(self):
        return f'({self.name}, {self.type})'

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name


def dijkstras(env):
    grid = env.grid
    s = str(env)
    i = 0
    row = col = 0
    node_id = 0
    V = [] # 2d grid of nodes
    v = [] # hold each row of nodes

    start = end = None

    while i < len(s):
        if s[i] == '\n':
            i += 1
            col += 1
            row = 0
            if len(v) > 0:
                V.append(v)
                v = []
        else:
            node = s[i:i+2]
            if node in ['  ', '>>', 'GG', 'WG']:
                if node == '  ':
                    node_type = 'floor'
                elif node == '>>':
                    node_type = 'start'
                elif node == 'GG':
                    node_type = 'end'
                elif node == 'WG':
                    node_type = 'wall'
                else:
                    raise NotImplementedError(node)
                # create new node
                n = Node(node_id, node_type, row, col)
                if node_type == 'start':
                    start = n
                elif node_type == 'end':
                    end = n
                v.append(n)
                node_id += 1

            i += 2
            row += 1

    if len(v) > 0: # get last row
        V.append(v)

    assert len(V) == grid.height and all([len(v) == grid.width for v in V])
    # create adj list
    for i, row in enumerate(V):
        for j, node in enumerate(row):
            if node.type in ['floor', 'start', 'end']: # agent can be here
                if i+1 < grid.height and V[i+1][j].type != 'wall':
                    down_node = V[i+1][j]
                    # create edge
                    node.adj.append(down_node)
                    down_node.adj.append(node) # undirected edge so both ways

                if j+1 < grid.width and V[i][j+1].type != 'wall':
                    right_node = V[i][j+1]
                    node.adj.append(right_node)
                    down_node.adj.append(right_node) # undirected edge so both ways

    # now run dijkstra's
    g = defaultdict(list)
    for row in V:
        for node in row:
            g[node] = [(1, other) for other in node.adj]

    visited = set()
    q = [(0, start, ())]
    dist = {start: 0}

    while q:
        cost, v1, path = heappop(q)
        if v1 not in visited:
            visited.add(v1)
            path = (v1, path)

            if v1 == end:
                return cost, path

            for c, v2 in g.get(v1, []):
                if v2 in visited:
                    continue
                prev = dist.get(v2, None)
                next = cost + c
                if prev is None or next < prev:
                    dist[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")



parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-Empty-5x5-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)

args = parser.parse_args()

env = gym.make(args.env)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

print(dijkstras(env))
window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
