# improved_program.py
import sys
from io import StringIO

def main(input_text: str) -> str:
    out_buf, orig_stdout = StringIO(), sys.stdout
    sys.stdout = out_buf
    try:
        inp = StringIO(input_text)
        def readints(): return list(map(int, inp.readline().split()))

        B, L, D = readints()
        book_scores = readints()

        libs = []
        for lib_id in range(L):
            N, T, M = readints()
            books = readints()
            # sort books in descending score once
            books.sort(key=lambda b: -book_scores[b])
            # maximum books this library could ever scan if signed-up first
            max_books = min(N, max(0, D - T) * M)
            potential = sum(book_scores[b] for b in books[:max_books])
            # value per signup day (tiny ε guards against T=0, though spec says T≥1)
            libs.append((-(potential / (T or 1e-9)), lib_id, T, M, books))

        # rank libraries by descending heuristic (negated for Python’s min-heap style sort)
        libs.sort()

        day = 0
        used = set()
        plan = []

        for _, lib_id, T, M, books in libs:
            if day + T >= D:
                continue  # no time to finish signup
            remaining_days = D - (day + T)
            capacity = remaining_days * M
            chosen = []

            for b in books:
                if b not in used:
                    chosen.append(b)
                    used.add(b)
                    if len(chosen) == capacity:
                        break

            if chosen:
                plan.append((lib_id, chosen))
                day += T  # start signup for next library after this one finishes

        # ---------- output ----------
        print(len(plan))
        for lib_id, chosen in plan:
            print(lib_id, len(chosen))
            print(*chosen)
    finally:
        sys.stdout = orig_stdout
    return out_buf.getvalue()
