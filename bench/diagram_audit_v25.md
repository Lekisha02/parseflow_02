# Diagram structure audit (v2.5)

How well does `processing/diagrams.py` read vector figures in a real textbook? Checked by hand against the rendered
figure, on *Introduction to Algorithms* (CLRS, 1,312 pages). Figures were drawn at random from the book's diagram
blocks with a fixed seed. Nothing here is a benchmark score. It is a small manual audit, and the counts are small.

Grades:

* **correct**: every node, edge, direction, label and value stated is right. Missing labels count against this
  grade.
* **mostly correct**: the structure is right, with one cosmetic slip, such as a row name missing a letter.
* **partial**: the structure is mostly right. Some labels are missing, garbled or misattributed, or an edge is
  missing or has lost its direction.
* **wrong**: the description would mislead a reader.

## Sample 1: development sample (12 figures, seed 7)

This sample was graded first, and the reader was then **fixed on it**. The "after" column describes this sample
after the fixes, so it is not an independent measurement.

| Figure (page) | Content | Before fixes | After fixes |
|---|---|---|---|
| p. 610 | MST graph, 9 vertices, 14 weighted edges | correct (14/14 edges and weights) | correct |
| p. 355 | red-black tree with a single sentinel; leaves point to it with curves | **wrong**: two invented edges (28–16, 28–19) | correct: 17 tree edges plus every NIL pointer to the sentinel; the root's curve to the sentinel is missing |
| p. 696 | flow network, multiple sources and sinks | partial: all edges and directions are right, 6 rotated weights are missing | same |
| p. 1192 | wavy paths between 6 vertices | **wrong**: one invented edge (y–v) | correct (6/6) |
| p. 199 | four heap states (MAX-HEAPIFY) | correct, including the heap-property check per state | correct |
| p. 238 | bucket sort: array and bucket lists | correct | correct |
| p. 991 | string-matching automaton and its table | partial: 13/13 edges, 4 labels missing, 1 label doubled ("a a") | 13/13 edges, 4 labels missing, none wrong |
| p. 281 | doubly linked list, 4 states | **wrong**: read as 3-cell arrays | correct: doubly linked, NIL at both ends, all keys in order |
| p. 644 | path s→x→y→u | partial: called a binary tree with right children | correct: a path, no sides |
| p. 918 | FFT recursion tree with subscripted labels | partial: tree right, labels garbled ("(a ,a ) 0 4") | correct ("(a_0,a_4)") |
| p. 1088 | logic circuit | **wrong**: gate shapes read as small graphs | not interpreted; the circuit is left to the image |
| p. 208 | quicksort partition arrays | partial: cells and colour groups right; the `p, i, j, r` markers and braces are not used | same |

## Sample 2: held-out sample (12 figures, seed 2026, drawn after the fixes above)

Two of the drawn figures (p. 355 and p. 572) were part of the earlier development set, so they are left out here.
That leaves **10 held-out figures**. Their grades below are for the code **before** this sample was looked at.

| Figure (page) | Content | Grade |
|---|---|---|
| p. 344 | BST search path | correct |
| p. 1105 | 3-CNF clause graph, 9 vertices with repeated names, 21 edges | correct: 21/21 edges, checked against the formula |
| p. 1189 | four graphs (subgraph / isomorphism figure) | correct: G checked edge by edge (9/9); the other three have the right edge counts |
| p. 1091 | machine-configuration block diagram | mostly correct: rows, cells and arrows are right; row names lost the letter c (c₀ became "0") |
| p. 643 | Dijkstra, 6 states | partial: weighted edges right, but highlight strokes added duplicate edges, one edge lost its direction, and one edge (y→t, 3) is missing |
| p. 544 | disjoint-set graph | partial: graph right; a vertical path e–f–g was called a binary tree with right children |
| p. 786 | merge-sort computation DAG | partial: arrows right, but most nodes are unlabelled circles and the block labels became nodes |
| p. 905 | FFT block diagram | partial: 4/4 edges and directions right; multi-line box labels garbled, 2 of 4 edge labels missing |
| p. 1177 | Venn diagrams | **wrong**: read as empty linked lists |
| p. 1081 | Venn diagrams (P, NP, co-NP) | **wrong**: overlapping ellipses read as a graph |

**Held-out result: 3 correct, 1 mostly correct, 4 partial, 2 wrong (out of 10).**

In the partial cases, the relationships that were stated were right, apart from the duplicates on p. 643 and the
binary-tree naming on p. 544. The errors were omissions: missing labels and unlabelled nodes. The two wrong
readings were both Venn diagrams.

The held-out sample then led to four fixes. Overlapping shapes are no longer read as nodes, and boxes with nothing
in them are no longer read as lists. Highlight copies of an edge are merged into one edge. A child straight below
its parent gets no left/right side.

After these fixes, p. 1177 and p. 1081 are no longer interpreted: they reach the model as images. On p. 643, 9 of
10 edges are right with their directions and weights. On p. 544, the path is no longer called binary. These
improvements were measured on the same sample, so they are not a held-out number.

## Whole book

After the fixes the book gives 153 diagram blocks (sequences 161, binary trees 90, other trees 94, graphs 81, directed graphs 106, linked-list groups 17, automata 1). Processing the whole book took 219 s (0.167 s/page) with a peak of 378 MB, in one process on 2 vCPUs. Figures that are not
vector drawings, and figures whose shapes carry no text, are left as figure blocks and are not given a structure.
