import tokenizer as t


def test_success_spacy_tokenizer():
    assert t.spacy_tokenizer(["hello,  world ", "heYa world."], tokens=True) == [
        ["hello", ",", " ", "world"],
        ["heYa", "world", "."],
    ]
    assert t.spacy_tokenizer(["hello,  world", "heYa world."], tokens=False) == [
        ["hello ,   world"],
        ["heYa world ."],
    ]
    assert t.spacy_tokenizer("hello,  wOrld ", tokens=True) == [
        "hello",
        ",",
        " ",
        "wOrld",
    ]
    assert t.spacy_tokenizer("hello,  wOrld", tokens=False) == ["hello ,   wOrld"]


def test_success_default_tokenizer():
    assert t.default_tokenizer(["hello,  world ", "heYa world."], tokens=True) == [
        ["hello", "world"],
        ["heya", "world"],
    ]
    assert t.default_tokenizer(["hello, world", "heYa world."], tokens=False) == [
        ["hello world"],
        ["heya world"],
    ]
    assert t.default_tokenizer("hello,  wOrld ", tokens=True) == ["hello", "world"]
    assert t.default_tokenizer("hello,  wOrld", tokens=False) == ["hello world"]


test_success_spacy_tokenizer()
test_success_default_tokenizer()
