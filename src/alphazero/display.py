"""
Formatting and disk I/O helpers
Based on `rich` package
"""

import json
from contextlib import nullcontext

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


# --- JSON I/O ---

def _to_jsonable(x):
    return x


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_jsonable)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --- Progress bar ---

def make_progress(console, verbose=True):
    """
    Iteration progress bar, disabled when not verbose
    """
    return Progress(
        TextColumn("[bold blue]iter {task.completed}/{task.total}"),
        BarColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console, disable=not verbose,
    )


def status(console, msg, verbose=True):
    """
    Spinner context manager, no-op when not verbose
    """
    return console.status(msg, spinner="dots") if verbose else nullcontext()


# --- Per-iteration ---

def print_iter(console, idx_iter, log, eval_metrics):
    """
    One block per iteration: train line, optional eval line
    eval_metrics=None -> no eval ran this iteration
    """
    log_nn = log["logs_nn_mean"]
    has_eval = eval_metrics is not None
    branch = "├─" if has_eval else "└─"

    console.print(
        f"[bold blue]iter {idx_iter:3d}[/bold blue]  [dim]tot {log['time_iter']:4.1f}s[/dim]"
    )
    console.print(
        f"[dim]{branch}[/dim] [green]train[/green]  "
        f"loss=[bold]{log_nn['loss']:.4f}[/bold]  "
        f"policy={log_nn['policy_loss']:.4f}  "
        f"value={log_nn['value_loss']:.4f}  "
        f"val_mean={log['value_mean']:.3f}  "
        f"[dim]|[/dim]  "
        f"[cyan]roll {log['time_roll']:4.1f}s[/cyan]  "
        f"[yellow]upd {log['time_updates']:4.1f}s[/yellow]"
    )
    if has_eval:
        console.print(
            f"[dim]└─[/dim] [green]eval [/green]  "
            f"succ=[bold green]{eval_metrics['success']:.2f}[/bold green]  "
            f"score={eval_metrics['score']:.3f}  "
            f"len={eval_metrics['length']:.1f}  "
            f"[dim]|[/dim]  "
            f"[magenta]eval {eval_metrics['time_eval']:4.1f}s[/magenta]"
        )
    console.print()


# --- One-off messages ---

def print_compiled(console, time_warm):
    console.print(f"[green]✓ Compiled in {time_warm:.1f}s")


def print_resumed(console, label, idx_iter, n_iter, n_logs, dir_checkpoint, n_gpu_ckpt, n_gpu_now):
    """
    Resume banner, warns when the device count changed since the checkpoint
    """
    if n_gpu_ckpt != n_gpu_now:
        console.print(
            f"[yellow]⚠ checkpoint trained on {n_gpu_ckpt} device(s), resuming on {n_gpu_now} "
            f"-- local minibatch sizes differ, training trajectory will not match[/yellow]"
        )
    console.print(
        f"[green]✓ Resumed[/green] from [bold]checkpoint_{label}[/bold]  "
        f"[dim]|[/dim]  iter=[bold]{idx_iter}[/bold]/{n_iter}  "
        f"[dim]|[/dim]  {n_logs} logs restored  "
        f"[dim]|[/dim]  {dir_checkpoint}"
    )


def print_dropped_samples(n_minibatch_local, N_local, size_minibatch, rem):
    """
    Soft warning from ShardingPlan: leftover samples dropped by the minibatch split
    """
    Console().print(
        f"[yellow]⚠ n_minibatch_local={n_minibatch_local} does not divide "
        f"N_local={N_local} (n_roll·n_step·n_batch/n_gpu).[/yellow]  "
        f"size_minibatch={size_minibatch}, "
        f"[bold]{rem}[/bold] samples dropped per update per device "
        f"([dim]{100 * rem / N_local:.2f}% of data[/dim])."
    )



