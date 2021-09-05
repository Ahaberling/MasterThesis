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
'''