import pandas as pd
import sys

from nltk.corpus import wordnet as wn


# https://www.nltk.org/howto/wordnet.html
def wup_similarity_depth(one, other, verbose=False, simulate_root=True):
    """
    Based on wordnet Wu-Palmer Similarity
    Wu-Palmer Similarity:
    Return a score denoting how similar two word senses are, based on the
    depth of the two senses in the taxonomy and that of their Least Common
    Subsumer (most specific ancestor node). Previously, the scores computed
    by this implementation did _not_ always agree with those given by
    Pedersen's Perl implementation of WordNet Similarity. However, with
    the addition of the simulate_root flag (see below), the score for
    verbs now almost always agree but not always for nouns.

    The LCS does not necessarily feature in the shortest path connecting
    the two senses, as it is by definition the common ancestor deepest in
    the taxonomy, not closest to the two senses. Typically, however, it
    will so feature. Where multiple candidates for the LCS exist, that
    whose shortest path to the root node is the longest will be selected.
    Where the LCS has multiple paths to the root, the longer path is used
    for the purposes of the calculation.

    :type  other: Synset
    :param other: The ``Synset`` that this ``Synset`` is being compared to.
    :type simulate_root: bool
    :param simulate_root: The various verb taxonomies do not
        share a single root which disallows this metric from working for
        synsets that are not connected. This flag (True by default)
        creates a fake root that connects all the taxonomies. Set it
        to false to disable this behavior. For the noun taxonomy,
        there is usually a default root except for WordNet version 1.6.
        If you are using wordnet 1.6, a fake root will be added for nouns
        as well.
    :return: (depth,subsumer, similarity)
        - subsumer depth
        - subsumer
        - A float score denoting the similarity of the two ``Synset``
        objects, normally greater than zero. If no connecting path between
        the two senses can be found, None is returned.

    """
    need_root = one._needs_root() or other._needs_root()

    # Note that to preserve behavior from NLTK2 we set use_min_depth=True
    # It is possible that more accurate results could be obtained by
    # removing this setting and it should be tested later on
    subsumers = one.lowest_common_hypernyms(
        other, simulate_root=simulate_root and need_root, use_min_depth=True
    )

    # If no LCS was found return None
    if len(subsumers) == 0:
        return None

    subsumer = one if one in subsumers else subsumers[0]

    # Get the longest path from the LCS to the root,
    # including a correction:
    # - add one because the calculations include both the start and end
    #   nodes
    depth = subsumer.max_depth() + 1

    # Note: No need for an additional add-one correction for non-nouns
    # to account for an imaginary root node because that is now
    # automatically handled by simulate_root
    # if subsumer._pos != NOUN:
    #     depth += 1

    # Get the shortest path from the LCS to each of the synsets it is
    # subsuming.  Add this to the LCS path length to get the path
    # length from each synset to the root.
    len1 = one.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    len2 = other.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    if len1 is None or len2 is None:
        return None
    # print (len1, len2, depth)
    len1 += depth
    len2 += depth
    # print (len1, len2, depth, (2.0 * depth) / (len1 + len2))
    simi = (2.0 * depth) / (len1 + len2)
    return depth, subsumer, simi
