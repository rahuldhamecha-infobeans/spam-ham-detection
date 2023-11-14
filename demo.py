def text_analysis(paragraph):
    weak_words = [
        'Absolutely', 'Definitely', 'Totally', 'Actually', 'Personally', 'Technically',
        'Virtually', 'Simply', 'Possibly', 'Somehow', 'Just', 'Very', 'Pretty', 'Some', 'Honestly',
        'That', 'Extremely', 'Really', 'Much', 'Exactly', 'Ultimate', 'Complete', 'World-class',
        'Amazing'
    ]

    filler_words = [
        'Well', 'um', 'uh', 'umm', 'hmm', 'hmmm', 'Like', 'Basically', 'seriously',
        'literally', 'totally', 'Clearly', 'you see', 'you know', 'I mean', 'You know what I mean',
        'At the end of the day', 'Believe me', 'Okay'
    ]

    founded_weak_words = []
    founded_filler_words = []
    paragraph_lower = paragraph.lower()

    for word in weak_words:
        lower_word = word.lower()
        count = paragraph_lower.count(lower_word)
        if int(count) > 0:
            data = {'word': word, 'count': count}
            founded_weak_words.append(data)

    for word in filler_words:
        lower_word = word.lower()
        count = paragraph_lower.count(lower_word)
        if int(count) > 0:
            data = {'word': word, 'count': count}
            founded_filler_words.append(data)

    return {'weak_words': founded_weak_words, 'filler_words': founded_filler_words}
