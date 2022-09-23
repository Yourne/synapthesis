def strjoin(separator, words):
    assert type(words) is list
    assert type(separator) is str
    s = words[0]
    for w in words[1:]:
        s += separator + w
    return s
