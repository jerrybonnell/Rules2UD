
def test_sent_not_corrupted(sent, out):
    # cross-check itself to make sure column[1] == sent
    check_sent = "".join([out[i][1] for i in range(1, len(out) - 2)])
    # print(f"answer : [{sent}]")
    # print(f"ours   : [{check_sent}]")
    if check_sent != sent:
        return False
    are_blank = [len(out[i][1]) == 0 for i in range(1, len(out) - 2)]
    if sum(are_blank) != 0:
        return False
    return True

def sent_not_corrupted(sent):
    import contextlib
    import sys
    import os

    class DummyFile(object):
        def write(self, x): pass
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout


    sys.path.append('./')
    import modify_conllu as m
    with nostdout():
        out = m.modify_conllu(sent, model="esupar")

    return test_sent_not_corrupted(sent, out)
