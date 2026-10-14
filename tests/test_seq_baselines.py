"""GRU4Rec and BERT4Rec on the shared recipe (S4, 0-B).

GRU4Rec: PAD never enters the recurrence, causality, SASRec's labels and loss code.
BERT4Rec: the target is never an input (eval or training), hidden slots are really hidden,
the history window is the other models', seeded masks, the cloze loss by definition.
"""
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mvcp import cli
from mvcp.compare import compare
from mvcp.data import negatives as neg
from mvcp.data.dataset import CachedCollator, SequenceCollator, UserSeq
from mvcp.eval.harness import evaluate
from mvcp.models.registry import build_model
from mvcp.models.sasrec import SASRecCollator
from mvcp.train.loop import TrainConfig, train_model

from conftest import ROOT
from test_cli import _run, tiny                    # noqa: F401  (fixture)
from test_instacart import _users
from test_instacart_all import synth               # noqa: F401  (fixture)

EMPTY = UserSeq(99, [], [], 3, False, False)


def _build(name, L=16, **kw):
    torch.manual_seed(0)
    model, col = build_model(name, num_items=100, dim=32, max_seq_len=L, max_num_txn=8, **kw)
    model.eval()
    return model, col


def _scores(model, x):
    return (x @ model.item_embedding.weight[1:].t())


def _memorises(name, **kw):
    """Loss on a small fixed set of users falls far below the uniform level ln(N)."""
    users = _users(n=24, num_items=40, seed=1)
    torch.manual_seed(0)
    model, col = build_model(name, 40, dim=32, max_seq_len=24, max_num_txn=8, dropout=0.0,
                             **kw)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = col(users)
    first = None
    for _ in range(300):
        loss = model.train_loss(batch)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return first, loss.item()


# ---------------------------------------------------------------- GRU4Rec

def _gru_hidden(model, batch):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, batch)


def test_gru_left_padding_does_not_change_scores_or_loss(toy_user, make_seq):
    """No position table and PAD never enters the GRU: a wider padded window gives the
    same states at the real positions, the same scores and the same loss."""
    model, col = _build("gru4rec", L=16)
    users = [toy_user, make_seq(1, [[5]], 6), make_seq(2, [[1, 2], [3]], 4)]
    b16, b32 = col(users), SASRecCollator(32, 8)(users)
    h16, h32 = _gru_hidden(model, b16), _gru_hidden(model, b32)
    assert torch.allclose(h16[~b16["pad_mask"]], h32[~b32["pad_mask"]], atol=1e-6)
    assert (h32[b32["pad_mask"]] == 0).all()
    with torch.no_grad():
        assert torch.allclose(model(b16), model(b32), atol=1e-6)
        assert torch.allclose(model.train_loss(b16), model.train_loss(b32), atol=1e-6)
    # a batch-mate's length does not matter either (rows are packed by their own length)
    alone = _gru_hidden(model, col([users[1]]))
    assert torch.allclose(alone[0, -1], h16[1, -1], atol=1e-6)


def test_gru_packed_path_equals_stepping_each_user_alone():
    """An independent path: every user's real tokens fed one at a time to a GRUCell that
    carries the layer's weights, from a zero state -- no padding, rolling or packing."""
    model, col = _build("gru4rec", L=24)
    enc = model.encoder
    cell = torch.nn.GRUCell(32, 32)
    cell.load_state_dict({k.replace("_l0", ""): v for k, v in enc.gru.state_dict().items()})
    users = _users(n=12)
    batch = col(users)
    h = _gru_hidden(model, batch)
    with torch.no_grad():
        for b, u in enumerate(users):
            x = enc.input_norm(model.item_embedding(torch.tensor(u.items[-24:]) + 1))
            state = torch.zeros(1, 32)
            for j, x_t in enumerate(x):
                state = cell(x_t[None], state)
                assert torch.allclose(h[b, 24 - len(x) + j], state[0], atol=1e-6)


def test_gru_pad_tokens_never_enter_the_recurrence(toy_user):
    model, col = _build("gru4rec")
    batch = col([toy_user])
    h = _gru_hidden(model, batch)
    garbage = dict(batch, item_tokens=batch["item_tokens"].masked_fill(batch["pad_mask"], 77))
    assert torch.isfinite(h).all()
    assert torch.equal(h, _gru_hidden(model, garbage))


def test_gru_is_causal(make_seq):
    model, col = _build("gru4rec")
    a = _gru_hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)]))
    b = _gru_hidden(model, col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)]))
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)   # all earlier positions
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-4)  # newest one does change
    # ... and it does read the order (SASRec's total-order assumption, kept on purpose)
    c = _gru_hidden(model, col([make_seq(0, [[11, 10, 12], [20], [30, 31]], 40)]))
    assert not torch.allclose(a[:, -1], c[:, -1], atol=1e-4)


def test_gru_uses_sasrec_collator_and_labels(toy_user):
    _, col = _build("gru4rec")
    _, sas = _build("sasrec")
    assert type(col) is SASRecCollator
    want, got = sas([toy_user]), col([toy_user])
    assert want.keys() == got.keys() and all(torch.equal(want[k], got[k]) for k in want)
    assert got["seq_labels"][0, -6:].tolist() == [11, 12, 20, 30, 31, 40]


def test_gru_readout_is_newest_state_and_empty_history_is_finite(toy_user):
    model, col = _build("gru4rec")
    batch = col([toy_user, EMPTY])
    h = _gru_hidden(model, batch)
    with torch.no_grad():
        assert torch.equal(model.user_embedding(batch), h[:, -1])
        assert torch.isfinite(model(batch)).all()
    model.train()
    loss = model.train_loss(batch)                    # the empty row has no label
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", ["gru4rec", "gru4rec_final2"])
def test_gru_compact_loss_equals_dense(name):
    users = _users() + [EMPTY]
    torch.manual_seed(0)
    dense, col = build_model(name, 40, dim=16, max_seq_len=8, max_num_txn=4)
    torch.manual_seed(0)
    compact, _ = build_model(name, 40, dim=16, max_seq_len=8, max_num_txn=4,
                             compact_loss=True, len_split=True)   # len_split: no effect
    dense.eval(), compact.eval()
    batch = col(users)
    a, b = dense.train_loss(batch), compact.train_loss(batch)
    assert torch.allclose(a, b, atol=1e-6)
    ga = torch.autograd.grad(a, dense.item_embedding.weight)[0]
    gb = torch.autograd.grad(b, compact.item_embedding.weight)[0]
    assert torch.allclose(ga, gb, atol=1e-6)


def test_gru_final2_weights_newest_position(toy_user, make_seq):
    base, col = _build("gru4rec")
    m, _ = _build("gru4rec_final2")
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in base.state_dict().items())
    b = col([toy_user, make_seq(1, [], 5)])
    with torch.no_grad():
        assert torch.equal(m.user_embedding(b), base.user_embedding(b))
        lab = b["seq_labels"]
        ce = F.cross_entropy(_scores(m, _gru_hidden(m, b)).flatten(0, 1), lab.flatten(),
                             ignore_index=-100, reduction="none").view_as(lab)
        real = ce[0, -6:]
        assert torch.allclose(m.train_loss(b), (real[:-1].sum() + 2 * real[-1]) / 7,
                              atol=1e-6)
        assert torch.allclose(base.train_loss(b), real.mean(), atol=1e-6)


def test_gru_memorises_small_data():
    first, last = _memorises("gru4rec")
    assert first > 3.5 and last < 0.5              # ln(40) = 3.69


# ---------------------------------------------------------------- BERT4Rec

def _bert_hidden(model, batch, masked=None):
    with torch.no_grad():
        return model.encoder.hidden_states(model.item_embedding, batch, masked)


def _captured_input(model, batch, masked=None):
    """What the transformer actually reads: (x, pos, pad) of the extended sequence."""
    seen = {}
    real = model.encoder._encode
    model.encoder._encode = lambda ext: seen.update(ext) or real(ext)
    try:
        _bert_hidden(model, batch, masked)
    finally:
        del model.encoder._encode
    return seen["x"], seen["pos"], seen["pad_mask"]


def test_bert_eval_scores_do_not_depend_on_the_target(make_seq):
    model, col = _build("bert4rec")
    txns = [[10, 11, 12], [20], [30, 31]]
    a, b = col([make_seq(0, txns, 40)]), col([make_seq(0, txns, 77)])
    with torch.no_grad():
        assert torch.equal(model(a), model(b))
    # training too: the target is a label, never an input
    masked = torch.zeros_like(a["pad_mask"])
    assert torch.equal(_bert_hidden(model, a, masked), _bert_hidden(model, b, masked))
    assert not torch.allclose(model.train_loss(a, masked), model.train_loss(b, masked))


def test_bert_eval_input_is_history_plus_one_mask_slot(toy_user):
    model, col = _build("bert4rec", L=16)
    batch = col([toy_user, EMPTY])
    x, pos, pad = _captured_input(model, batch)
    emb = model.item_embedding(batch["item_tokens"])
    assert x.shape == (2, 17, 32)
    assert torch.equal(x[:, :-1], emb)                         # the history, untouched
    assert torch.equal(x[:, -1], model.encoder.mask_emb.expand(2, -1))
    assert pos[0, -7:].tolist() == [1, 2, 3, 4, 5, 6, 7] and pos[1, -1] == 1
    assert not pad[:, -1].any() and torch.equal(pad[:, :-1], batch["pad_mask"])
    h = _bert_hidden(model, batch)
    with torch.no_grad():
        assert torch.equal(model.user_embedding(batch), h[:, -1])
        assert torch.isfinite(model(batch)).all()              # empty history: [MASK] alone


def test_bert_states_equal_each_user_encoded_alone_and_train_path(toy_user, make_seq):
    """No padding at all (one user, window = its own length) gives the batch's states; the
    no-grad eval pass (the encoder's fast path) and the training-mode pass agree."""
    model, col = _build("bert4rec", L=16, dropout=0.0)
    users = [toy_user, make_seq(1, [[5]], 6), make_seq(2, [[1, 2], [3]], 4)]
    h = _bert_hidden(model, col(users))
    for b, u in enumerate(users):
        n = len(u.items)
        alone = _bert_hidden(model, SequenceCollator(n, 8)([u]))
        assert alone.shape[1] == n + 1
        assert torch.allclose(h[b, -(n + 1):], alone[0], atol=1e-6)
    model.train()                                      # dropout 0: the regular code path
    with torch.enable_grad():
        slow = model.encoder.hidden_states(model.item_embedding, col(users))
    real = F.pad(~col(users)["pad_mask"], (0, 1), value=True)
    assert torch.allclose(h[real], slow[real], atol=1e-5)


def test_bert_history_window_is_the_other_models(make_seq):
    """L history items + the [MASK] slot: nothing is dropped to make room for it."""
    L = 16
    model, col = _build("bert4rec", L=L)
    _, sas = _build("sasrec", L=L)
    long = make_seq(0, [[i] for i in range(1, 21)], 50)        # 20 items > L
    batch = col([long])
    assert type(col) is SequenceCollator
    assert torch.equal(batch["item_tokens"], sas([long])["item_tokens"])
    assert (batch["item_tokens"][0] - 1).tolist() == list(range(5, 21))
    assert model.encoder.pos_emb.num_embeddings == L + 2       # PAD + 1..L + the slot
    _, pos, _ = _captured_input(model, batch)
    assert pos[0].tolist() == list(range(1, L + 2))
    assert _bert_hidden(model, batch).shape == (1, L + 1, 32)


def test_bert_hidden_slot_does_not_see_its_own_item(make_seq):
    model, col = _build("bert4rec")
    a = col([make_seq(0, [[10, 11, 12], [20], [30, 31]], 40)])
    b = col([make_seq(0, [[10, 11, 12], [55], [30, 31]], 40)])   # differs at one slot
    masked = torch.zeros_like(a["pad_mask"])
    masked[0, -3] = True                                          # ... which is hidden
    assert torch.equal(_bert_hidden(model, a, masked), _bert_hidden(model, b, masked))
    assert not torch.allclose(_bert_hidden(model, a), _bert_hidden(model, b), atol=1e-4)
    # bidirectional: a later item changes an earlier slot's state (SASRec's would not)
    c = col([make_seq(0, [[10, 11, 12], [20], [30, 55]], 40)])
    assert not torch.allclose(_bert_hidden(model, a)[0, -7], _bert_hidden(model, c)[0, -7],
                              atol=1e-4)


def test_bert_left_padding_does_not_change_scores_or_loss(toy_user, make_seq):
    model, col = _build("bert4rec", L=32)
    users = [toy_user, make_seq(1, [[5]], 6)]
    b16, b32 = SequenceCollator(16, 8)(users), col(users)
    m16 = torch.zeros_like(b16["pad_mask"])
    m16[0, -2] = m16[0, -5] = m16[1, -1] = True
    m32 = F.pad(m16, (16, 0))
    h16, h32 = _bert_hidden(model, b16, m16), _bert_hidden(model, b32, m32)
    assert torch.allclose(h16[:, -7:], h32[:, -7:], atol=1e-6)
    with torch.no_grad():
        assert torch.allclose(model(b16), model(b32), atol=1e-6)
        assert torch.allclose(model.train_loss(b16, m16), model.train_loss(b32, m32),
                              atol=1e-6)
    garbage = dict(b32, item_tokens=b32["item_tokens"].masked_fill(b32["pad_mask"], 77))
    real = F.pad(~b32["pad_mask"], (0, 1), value=True)            # + the [MASK] slot
    assert torch.allclose(h32[real], _bert_hidden(model, garbage, m32)[real], atol=1e-6)


def test_bert_mask_is_seeded_and_only_on_real_slots():
    model, col = _build("bert4rec", mask_prob=0.5)
    batch = col(_users(n=12))

    def draw(seed):
        torch.manual_seed(seed)
        return model.draw_mask(batch)
    assert torch.equal(draw(3), draw(3)) and not torch.equal(draw(3), draw(4))
    assert not (draw(3) & batch["pad_mask"]).any()
    torch.manual_seed(3)
    first, second = model.draw_mask(batch), model.draw_mask(batch)   # redrawn every step
    assert not torch.equal(first, second)
    model.mask_prob = 1.0
    assert torch.equal(draw(0), ~batch["pad_mask"])
    # train_loss draws it itself, from the torch RNG
    model.mask_prob = 0.5
    torch.manual_seed(3)
    a = model.train_loss(batch)
    torch.manual_seed(3)
    assert torch.equal(a, model.train_loss(batch))
    assert torch.equal(a, model.train_loss(batch, draw(3)))


def test_bert_cached_collate_rows_carry_no_mask():
    model, col = _build("bert4rec", mask_prob=0.5)
    users = _users(n=12)
    cached = CachedCollator(col)
    want = col(users)
    for _ in range(2):
        torch.manual_seed(0)
        model.train_loss(cached(users))
        got = cached(users)
        assert all(torch.equal(want[k], got[k]) for k in want)


def test_bert_mask_prob_zero_leaves_only_the_target_term():
    model, col = _build("bert4rec", mask_prob=0.0)
    batch = col(_users(n=6) + [EMPTY])
    assert not model.draw_mask(batch).any()
    with torch.no_grad():
        want = F.cross_entropy(model(batch), batch["target"])
        assert torch.allclose(model.train_loss(batch), want, atol=1e-6)


@pytest.mark.parametrize("name,fw", [("bert4rec", 1.0), ("bert4rec_final2", 2.0)])
def test_bert_loss_is_ce_over_hidden_slots(name, fw):
    base, _ = _build("bert4rec")
    model, col = _build(name)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in base.state_dict().items())
    batch = col(_users(n=5))
    torch.manual_seed(1)
    masked = model.draw_mask(batch)
    n_hist = int(masked.sum())
    assert 0 < n_hist < int((~batch["pad_mask"]).sum())
    h = _bert_hidden(model, batch, masked)
    with torch.no_grad():
        assert torch.equal(model(batch), base(batch))            # eval path: no FW
        hist = F.cross_entropy(_scores(model, h[:, :-1][masked]),
                               batch["item_tokens"][masked] - 1, reduction="sum")
        tgt = F.cross_entropy(_scores(model, h[:, -1]), batch["target"], reduction="sum")
        want = (hist + fw * tgt) / (n_hist + fw * 5)
        assert torch.allclose(model.train_loss(batch, masked), want, atol=1e-6)


@pytest.mark.parametrize("name", ["bert4rec", "bert4rec_final2"])
def test_bert_len_split_matches_single_pass(name):
    users = _users(n=13, seed=3) + [EMPTY]
    kw = dict(dim=16, max_seq_len=24, max_num_txn=8, mask_prob=0.3)
    torch.manual_seed(0)
    whole, col = build_model(name, 40, **kw)
    torch.manual_seed(0)
    split, _ = build_model(name, 40, len_split=True, compact_loss=True, **kw)
    whole.eval(), split.eval()
    for chunk in (users, users[:1], users[3:5], [users[2]] * 4):   # incl. B=1, equal lengths
        batch = col(chunk)
        torch.manual_seed(1)
        masked = whole.draw_mask(batch)
        real = torch.cat([~batch["pad_mask"], torch.ones(len(chunk), 1, dtype=torch.bool)], 1)
        h1, h2 = _bert_hidden(whole, batch, masked), _bert_hidden(split, batch, masked)
        assert torch.allclose(h1[real], h2[real], atol=1e-6)
        with torch.no_grad():
            assert torch.allclose(whole(batch), split(batch), atol=1e-6)
        a, b = whole.train_loss(batch, masked), split.train_loss(batch, masked)
        assert torch.allclose(a, b, atol=1e-6)
        ga = torch.autograd.grad(a, whole.item_embedding.weight)[0]
        gb = torch.autograd.grad(b, split.item_embedding.weight)[0]
        assert torch.allclose(ga, gb, atol=1e-6)


def test_bert_memorises_small_data():
    first, last = _memorises("bert4rec", mask_prob=0.3)
    assert first > 3.5 and last < 0.7              # ln(40) = 3.69


# ---------------------------------------------------------------- both: train loop, GPU

@pytest.mark.parametrize("name", ["gru4rec", "gru4rec_final2", "bert4rec", "bert4rec_final2"])
def test_train_eval_smoke(name):
    from test_train_smoke import _synth
    num_items = 30
    tr, va, te = (_synth(n, num_items, seed=s) for n, s in ((60, 1), (20, 2), (20, 3)))
    cfg = TrainConfig(lr=1e-2, epochs=4, batch_size=16, eval_every=2, patience=3, seed=0)
    infos = []
    for _ in range(2):                                 # the same seed: the same run
        torch.manual_seed(0)
        model, col = build_model(name, num_items, dim=16, max_seq_len=16, max_num_txn=8)
        infos.append(train_model(model, col, tr, va,
                                 neg.build(va, num_items, 0, "corrected", n=10),
                                 num_items, cfg))
    assert np.isfinite(infos[0]["best_val_score"]) and infos[0] == infos[1]
    model.eval()
    res = evaluate(model, te, col, neg.build(te, num_items, 0, "corrected", n=10),
                   num_items, ks=(1, 5, 10))
    assert res["n_users"] == 20


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("name", ["gru4rec_final2", "bert4rec_final2"])
def test_gpu_training_is_reproducible(name):
    """The cuDNN GRU path (packed sequences) and the CUDA mask draws repeat exactly."""
    from mvcp.utils.seed import seed_everything
    users = {u.user_idx: u for u in _users(n=48, num_items=200, seed=5)}
    states = []
    # as the CLI does for the long-sequence datasets (data yaml `sdp_mem_efficient: false`)
    prev = torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    try:
        for _ in range(2):
            seed_everything(7)
            model, col = build_model(name, 200, dim=32, max_seq_len=24, max_num_txn=8,
                                     compact_loss=True, len_split=True)
            cfg = TrainConfig(lr=1e-3, epochs=3, batch_size=16, eval_every=1, patience=3,
                              seed=7, device="cuda")
            info = train_model(model, col, users, users,
                               neg.build(users, 200, 0, "corrected", n=10), 200, cfg)
            states.append((info, {k: v.cpu() for k, v in model.state_dict().items()}))
    finally:
        torch.backends.cuda.enable_mem_efficient_sdp(prev)
    assert states[0][0] == states[1][0]
    assert all(torch.equal(v, states[1][1][k]) for k, v in states[0][1].items())


# ---------------------------------------------------------------- CLI + tools end to end

def _name(argv):
    return cli._result_name(cli._train_parser().parse_args(argv))


def _script(name, *args):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *args], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


BASELINES = (("gru4rec", ()), ("gru4rec_final2", ()),
             ("bert4rec", ("--mask-prob", "0.4")), ("bert4rec_final2", ("--mask-prob", "0.4")))
HP = ("--dropout", "0.1", "--lr", "0.001")


def test_result_name_mask_prob(monkeypatch):
    assert _name(["--model", "gru4rec", "--lr", "0.001", "--dim", "64", "--dropout", "0.1",
                  "--epochs", "500", "--eval-split", "val"]) == "gru4rec_lr0.001_d64_do0.1_val"
    assert _name(["--model", "bert4rec_final2", "--lr", "0.0005", "--dim", "128",
                  "--dropout", "0.3", "--mask-prob", "0.2"]) == \
        "bert4rec_final2_lr0.0005_d128_do0.3_mp0.2"
    assert _name(["--model", "bert4rec", "--mask-prob", "0.4", "--eval-split", "val"]) == \
        "bert4rec_mp0.4_val"
    monkeypatch.setattr(sys, "argv", ["mvcp-train", "--model", "sasrec", "--mask-prob", "0.4"])
    with pytest.raises(SystemExit):
        cli.train()


def test_cli_katchers_path_end_to_end(monkeypatch, tiny):
    """The Katchers path (CSV bundle, dense loss): the four names train on val only,
    --epochs overrides the yaml cap, and the compare / grid / expansion tools read them."""
    def train(model, seed, *extra):
        _run(monkeypatch, ["--model", model, "--seed", str(seed), "--eval-split", "val",
                           "--epochs", "3", "--dim", "32", "--out", str(tiny.out),
                           *HP, *extra])
    val = cli._bundle().seqs["val"]
    for u in list(val)[:3]:                 # the fixture's val split has no repeat user
        val[u] = replace(val[u], target=val[u].items[0], repeat=True)
    for seed in (0, 1):
        for model, extra in BASELINES:
            train(model, seed, *extra)
        train("bert4rec", seed, "--mask-prob", "0.2")
    assert set(tiny.seen) == {"val"}
    name = "_lr0.001_d32_do0.1"
    res = json.loads((tiny.out / f"bert4rec{name}_mp0.4_val_seed0.json").read_text())
    assert res["history"][-1]["epoch"] == 3                # the fixture's yaml says 2
    assert res["eval_split"] == "val" and "data" not in res
    assert res["full_catalog"]["overall"]["ndcg@10"] == res["best_val"]
    other = json.loads((tiny.out / f"bert4rec{name}_mp0.2_val_seed0.json").read_text())
    assert other["history"] != res["history"]              # mask_prob reaches the model

    r = compare(tiny.out, f"gru4rec{name}_val", f"bert4rec{name}_mp0.4_val", [0, 1], n_boot=50)
    assert r["seeds"] == [0, 1] and {x["stratum"] for x in r["rows"]} >= {"overall", "copurchase"}

    def masked(tag):
        return np.mean([json.loads((tiny.out / f"{tag}_val_seed{s}.json").read_text())
                        ["full_catalog"]["overall"]["ndcg@10"] for s in (0, 1)])
    grid = _script("report_ic_grid.py", "--dir", str(tiny.out), "--seeds", "0-1", "--protocol",
                   "full_catalog", "--models", "gru4rec,bert4rec")
    # mask_prob is part of the config key: two bert4rec rows
    assert all(f"| {m} | 32 | 0.1 | 0.001 | 2 | " in grid
               for m in ("gru4rec", "bert4rec_mp0.4", "bert4rec_mp0.2"))
    assert f"- best gru4rec: dim 32, dropout 0.1, lr 0.001 → {masked(f'gru4rec{name}'):.5f}" \
        in grid

    queue = _script("bl_expand_queue.py", "--data", "katchers", "--dir", str(tiny.out),
                    "--seeds", "0-1", "--mask-prob", "0.4")
    jobs = [ln for ln in queue.splitlines() if ln and not ln.startswith("#")]
    # dim 32 is the bottom of its ladder: 5 unseen neighbours per model and seed
    want = {(64, "0.1", "0.001"), (32, "0", "0.001"), (32, "0.3", "0.001"),
            (32, "0.1", "0.0005"), (32, "0.1", "0.002")}
    assert sorted(jobs) == sorted(
        f"{m} {s} --dim {d} --dropout {p} --lr {lr}{mp} --epochs 500 --eval-split val"
        for m, mp in (("gru4rec", ""), ("bert4rec", " --mask-prob 0.4"))
        for s in (0, 1) for d, p, lr in want)
    assert f"# bert4rec current best dim 32 / dropout 0.1 / lr 0.001 = " \
           f"{masked(f'bert4rec{name}_mp0.4'):.5f} (1 seen)" in queue
    # a seen neighbour is not queued again; its opposite side gets a boundary note
    train("gru4rec", 0, "--dropout", "0.3")
    train("gru4rec", 1, "--dropout", "0.3")
    queue = _script("bl_expand_queue.py", "--data", "katchers", "--dir", str(tiny.out),
                    "--seeds", "0-1", "--models", "gru4rec")
    jobs = [ln for ln in queue.splitlines() if ln and not ln.startswith("#")]
    assert len(jobs) == 2 * 4 and not any(ln.startswith("bert4rec") for ln in jobs)
    assert "gain of the last step in that direction" in queue


@pytest.fixture
def few_threads():
    """A CPU GRU on tiny matrices is ~50x slower with one thread per core."""
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prev)


def test_cli_instacart_all_path_end_to_end(synth, monkeypatch, tmp_path, few_threads):
    """--data instacart_all on the synthetic package (cached collate, compact loss, length
    split): strata and R5 reach the result files and the report tools pair the baselines
    with a model already in the folder."""
    real_load = cli.load_config

    def fake_load(p):
        cfg = real_load(p)
        if "configs/data/instacart_" in p:
            cfg["data_dir"] = str(synth["tmp"] / p.split("instacart_")[1].split(".")[0])
            cfg["max_seq_len"] = 64
        if "train" in p:
            cfg.update(nhead=2, num_layers=1, epochs=2, batch_size=16, device="cpu")
        return cfg
    monkeypatch.setattr(cli, "load_config", fake_load)
    out = tmp_path / "all"
    for seed in (0, 1):
        for model, extra in BASELINES + (("txf_txn_causal_final2", ()),):
            for split in ((), ("--eval-split", "val")):
                _run(monkeypatch, ["--model", model, "--data", "instacart_all", "--seed",
                                   str(seed), "--dim", "64", "--out", str(out),
                                   *HP, *extra, *split])
    name = "_lr0.001_d64_do0.1"
    for tag in (f"gru4rec_final2{name}", f"bert4rec_final2{name}_mp0.4"):
        res = json.loads((out / f"{tag}_seed0.json").read_text())
        st = res["full_nomask"]
        n = lambda k: st.get(k, {"n": 0})["n"]       # noqa: E731  (an empty stratum has no key)
        assert res["data"] == "instacart_all" and "r5" in res
        assert n("r1_in_last") + n("r1_before") == n("repeat") > 0
        val = json.loads((out / f"{tag}_val_seed0.json").read_text())
        assert val["full_nomask"]["overall"]["ndcg@10"] == val["best_val"]   # A5 selection

    pairs = ",".join(f"{lab}={tag}:txf_txn_causal_final2{name}" for lab, tag in
                     (("G", f"gru4rec_final2{name}"), ("B", f"bert4rec_final2{name}_mp0.4")))
    txt = _script("report_ic.py", "--seeds", "0-1", "--tag", "bl", "--n-boot", "50", "--dir",
                  str(out), "--strata", "overall,novel,repeat", "--detail", "--pairs", pairs)
    assert "#### Primary metric — unmasked full-catalog NDCG@10 overall (B − A)" in txt
    assert f"| B `bert4rec_final2{name}_mp0.4` → `txf_txn_causal_final2{name}` | " in txt
    res = json.loads((out / "compare" / "bl_G.json").read_text())
    assert res["primary"] == "full_nomask" and res["seeds"] == [0, 1]

    grid = _script("report_ic_grid.py", "--dir", str(out), "--seeds", "0-1",
                   "--models", "gru4rec,bert4rec,bert4rec_final2")
    assert all(f"| {m} | 64 | 0.1 | 0.001 | 2 | " in grid
               for m in ("gru4rec", "bert4rec_mp0.4", "bert4rec_final2_mp0.4"))
    queue = _script("bl_expand_queue.py", "--data", "instacart_all", "--dir", str(out),
                    "--seeds", "0-1", "--mask-prob", "0.4", "--models", "bert4rec")
    jobs = [ln for ln in queue.splitlines() if ln and not ln.startswith("#")]
    want = {(128, "0.1", "0.001"), (64, "0", "0.001"), (64, "0.3", "0.001"),
            (64, "0.1", "0.0005"), (64, "0.1", "0.002")}    # dim 64: the ladder's bottom
    assert sorted(jobs) == sorted(
        f"bert4rec {s} --data instacart_all --dim {d} --dropout {p} --lr {lr} "
        "--mask-prob 0.4 --eval-split val" for s in (0, 1) for d, p, lr in want)


def test_no_training_baselines_on_katchers_bundle(monkeypatch, tmp_path, capsys, make_seq):
    """scripts/instacart_baselines.py --data katchers reads the CSV bundle's UserSeqs."""
    spec = importlib.util.spec_from_file_location(
        "instacart_baselines", ROOT / "scripts" / "instacart_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)

    def users(lo, n):
        out = {}
        for u in range(lo, lo + n):
            txns = [[int(x) for x in rng.choice(60, size=int(rng.integers(1, 4)),
                                                replace=False)]
                    for _ in range(int(rng.integers(1, 5)))]
            # every third user re-buys an item of the newest order
            out[u] = make_seq(u, txns, txns[-1][0] if u % 3 == 0 else int(rng.integers(0, 60)))
        return out
    bundle = SimpleNamespace(seqs={"train": users(0, 40), "val": users(40, 12),
                                   "test": users(52, 12)},
                             num_items=60, test_negatives=None)
    monkeypatch.setattr(mod, "load_bundle", lambda d: bundle)
    monkeypatch.setattr(mod, "load_instacart", lambda *a, **k: pytest.fail("package path"))
    monkeypatch.setattr(mod, "load_config", lambda p: {"data_dir": "unused"})
    monkeypatch.setattr(sys, "argv", ["x", "--data", "katchers", "--out",
                                      str(tmp_path / "b.json"), "--primary", "full_catalog",
                                      "--strata", "copurchase"])
    mod.main()
    rows = json.loads((tmp_path / "b.json").read_text())
    assert "| masked copurchase |" in capsys.readouterr().out
    assert set(rows) >= {"val/pop/-", "test/persfreq/window", "test/histfirst/full"}
    assert rows["test/histfirst/full"]["full_catalog"]["repeat"]["ndcg@10"] == 1.0
    # the bundle's history is the model window: both views give the same numbers
    assert rows["test/persfreq/window"] == rows["test/persfreq/full"]
    # a re-bought item of the newest order: `lastfirst` ranks it above everything unseen
    assert rows["test/lastfirst/window"]["full_nomask"]["repeat"]["ndcg@10"] > \
        rows["test/pop/-"]["full_nomask"]["repeat"]["ndcg@10"]
