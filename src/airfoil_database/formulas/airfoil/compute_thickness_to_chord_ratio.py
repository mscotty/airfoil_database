# Compute Thickness-to-Chord Ratio
def thickness_to_chord_ratio(thickness, chord_length):
    """
    Compute the maximum thickness-to-chord ratio.
    
    @param thickness: Thickness distribution.
    @param chord_length: Chord length of the airfoil.
    @return: Maximum thickness-to-chord ratio.
    """
    if chord_length == 0:
        raise ZeroDivisionError("Chord length cannot be zero.")

    if len(thickness) == 0:
        raise ValueError("Thickness array cannot be empty.")

    return max(thickness) / chord_length