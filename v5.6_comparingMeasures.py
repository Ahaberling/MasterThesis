'''
1. get diffusion and recombination arrays from all three approaches + recombination dicts
2. diffusion:
    create list with all topics used in all three approaches combined
    for every array, extent them with this list so all have the same dimensions
3. recombination:
    create list with all recombinations used in all three approaches combined
    for every array, extent them with this list so all have the same dimensions

how often was threshhold reached?
how many counts?
cosine / jaccard similarity between arrays
are there columns identical, that are not sum = 0?
pick confident cases of all three approaches and compare them within approaches
argument: finding knowledge recombination is complex, so one plain measure is not enough.
            different typed of recombination are found in different ways. not clear yet
            in what way the recombinations differ

rename referenceMeasure to intuitiveMeasure

idea:
comparing diffusion should be finde

idea:
for recombination comparability focus first on recombination counts.
in order to compare recombination, the cD measure might be adpated by adding 1 to all 10 or 11 cells following a 1, vertically
look for patterns 111 111 111 0
if there are patterns like this in array[10:,:] then overthink this

'''