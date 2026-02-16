"""CLI entry point for face2face.

Commands:
    face2face client   — Start Machine A (proxy server + visual tx/rx)
    face2face server   — Start Machine B (forwarder + visual tx/rx)
    face2face calibrate — Interactive calibration wizard
    face2face benchmark — Test throughput of the visual channel
"""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
from pathlib import Path

import click

from .config import AppConfig, load_config, save_config


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", "-c", type=click.Path(), default=None,
              help="Path to config file")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Enable verbose/debug logging")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """face2face: HTTP proxy over visual channel (webcam/screen)."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--port", "-p", type=int, default=None,
              help="Proxy listen port (default: 8080)")
@click.option("--host", "-H", type=str, default=None,
              help="Proxy listen host (default: 127.0.0.1)")
@click.option("--fullscreen", "-f", is_flag=True, default=False,
              help="Display frames in fullscreen mode")
@click.option("--debug-capture", "-d", type=click.Path(), default=None,
              help="Save debug frames (raw, warped, grid) to this directory")
@click.option("--monitor", "-m", is_flag=True, default=False,
              help="Show a live webcam monitor window for camera alignment")
@click.pass_context
def client(ctx: click.Context, port: int | None, host: str | None,
           fullscreen: bool, debug_capture: str | None,
           monitor: bool) -> None:
    """Start the client (proxy server + visual transmitter/receiver).

    Run this on the machine that needs internet access.
    Point its webcam at the server machine's screen.

    Usage:
        face2face client
        export http_proxy=http://localhost:8080
        git clone http://github.com/user/repo
    """
    config: AppConfig = ctx.obj["config"]
    if port is not None:
        config.proxy_port = port
    if host is not None:
        config.proxy_host = host
    if fullscreen:
        config.fullscreen = True
    if debug_capture is not None:
        Path(debug_capture).mkdir(parents=True, exist_ok=True)
        config.debug_capture_dir = debug_capture
    if monitor:
        config.enable_monitor = True

    level = "DEBUG" if ctx.obj.get("verbose") else config.log_level
    _setup_logging(level)
    logger = logging.getLogger("face2face.client")

    async def run():
        from ..protocol.channel import VisualChannel
        from ..proxy.server import ProxyServer

        channel = VisualChannel(config.to_channel_config())
        proxy = ProxyServer(
            channel,
            host=config.proxy_host,
            port=config.proxy_port,
            timeout=config.proxy_timeout,
        )

        await channel.start()
        await proxy.start()

        logger.info(
            "Client ready. Set your proxy to http://%s:%d",
            config.proxy_host, config.proxy_port,
        )
        logger.info("Press Ctrl+C to stop.")

        stop_event = asyncio.Event()

        def handle_signal():
            stop_event.set()

        loop = asyncio.get_event_loop()
        try:
            if sys.platform != "win32":
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, handle_signal)
                await stop_event.wait()
            else:
                # Windows doesn't support add_signal_handler; wait with
                # periodic sleeps so KeyboardInterrupt can be raised.
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
        finally:
            logger.info("Shutting down...")
            await proxy.stop()
            await channel.stop()

    asyncio.run(run())


@cli.command()
@click.option("--fullscreen", "-f", is_flag=True, default=False,
              help="Display frames in fullscreen mode")
@click.option("--debug-capture", "-d", type=click.Path(), default=None,
              help="Save debug frames (raw, warped, grid) to this directory")
@click.option("--monitor", "-m", is_flag=True, default=False,
              help="Show a live webcam monitor window for camera alignment")
@click.pass_context
def server(ctx: click.Context, fullscreen: bool,
           debug_capture: str | None, monitor: bool) -> None:
    """Start the server (HTTP forwarder + visual transmitter/receiver).

    Run this on the machine with internet access.
    Point its webcam at the client machine's screen.
    """
    config: AppConfig = ctx.obj["config"]
    if fullscreen:
        config.fullscreen = True
    if debug_capture is not None:
        Path(debug_capture).mkdir(parents=True, exist_ok=True)
        config.debug_capture_dir = debug_capture
    if monitor:
        config.enable_monitor = True

    level = "DEBUG" if ctx.obj.get("verbose") else config.log_level
    _setup_logging(level)
    logger = logging.getLogger("face2face.server")

    async def run():
        from ..protocol.channel import VisualChannel
        from ..proxy.forwarder import ProxyForwarder

        channel = VisualChannel(config.to_channel_config())
        forwarder = ProxyForwarder(channel, timeout=config.proxy_timeout)

        await channel.start()
        await forwarder.start()

        logger.info("Server ready. Waiting for requests...")
        logger.info("Press Ctrl+C to stop.")

        stop_event = asyncio.Event()

        def handle_signal():
            stop_event.set()

        loop = asyncio.get_event_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, handle_signal)

        # Track fire-and-forget request handler tasks
        request_tasks: set[asyncio.Task] = set()

        # Main loop: dispatch incoming messages to forwarder
        async def dispatch_loop():
            while not stop_event.is_set():
                result = await channel.rx.get_message(timeout=1.0)
                if result is None:
                    continue
                msg_id, data = result
                # Each message gets its own stream for response routing
                stream_id = msg_id  # Use msg_id as stream_id for simplicity
                task = asyncio.create_task(
                    forwarder.handle_request(stream_id, data))
                request_tasks.add(task)
                task.add_done_callback(request_tasks.discard)

        dispatch_task = asyncio.create_task(dispatch_loop())

        try:
            if sys.platform != "win32":
                await stop_event.wait()
            else:
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
        finally:
            dispatch_task.cancel()
            try:
                await dispatch_task
            except asyncio.CancelledError:
                pass
            # Cancel in-flight request handlers so they release the session
            for task in request_tasks:
                task.cancel()
            if request_tasks:
                await asyncio.gather(*request_tasks, return_exceptions=True)
            logger.info("Shutting down...")
            await forwarder.stop()
            await channel.stop()

    asyncio.run(run())


@cli.command()
@click.option("--save", "-s", is_flag=True, default=False,
              help="Save calibration results to config file")
@click.pass_context
def calibrate(ctx: click.Context, save: bool) -> None:
    """Run the interactive calibration wizard.

    Tests the visual link and finds optimal parameters.
    """
    config: AppConfig = ctx.obj["config"]
    _setup_logging("INFO")

    from .calibrate import CalibrationWizard

    wizard = CalibrationWizard(config)
    try:
        updated = wizard.run()
        if save:
            save_config(updated, ctx.obj["config_path"])
            print(f"\nConfiguration saved to {ctx.obj['config_path'] or '~/.config/face2face/config.toml'}")
    finally:
        wizard.cleanup()


@cli.command()
@click.option("--duration", "-d", type=int, default=10,
              help="Benchmark duration in seconds")
@click.option("--loopback", "-l", is_flag=True, default=False,
              help="Use loopback channel (no hardware needed)")
@click.pass_context
def benchmark(ctx: click.Context, duration: int, loopback: bool) -> None:
    """Benchmark the visual channel throughput."""
    config: AppConfig = ctx.obj["config"]
    _setup_logging("WARNING")

    async def run():
        import time

        if loopback:
            from ..protocol.channel import LoopbackChannel
            channel = LoopbackChannel()
        else:
            from ..protocol.channel import VisualChannel
            channel = VisualChannel(config.to_channel_config())

        await channel.start()

        stream_id = await channel.allocate_stream()

        # Generate test data
        test_data = bytes(range(256)) * 4  # 1 KB chunks

        print(f"Benchmarking {'loopback' if loopback else 'visual'} channel "
              f"for {duration} seconds...")

        total_bytes = 0
        total_messages = 0
        start_time = time.monotonic()

        while time.monotonic() - start_time < duration:
            success = await channel.send(stream_id, test_data)
            if success:
                total_bytes += len(test_data)
                total_messages += 1

            if loopback:
                # Drain the loopback queue
                await channel.recv(stream_id, timeout=0.1)

        elapsed = time.monotonic() - start_time
        await channel.release_stream(stream_id)
        await channel.stop()

        throughput = total_bytes / elapsed if elapsed > 0 else 0
        print(f"\nResults:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Messages: {total_messages}")
        print(f"  Total data: {total_bytes:,} bytes")
        print(f"  Throughput: {throughput:.0f} bytes/s "
              f"({throughput * 8 / 1000:.1f} kbps)")

    asyncio.run(run())


@cli.command()
@click.option("--branch", "-b", default="claude/webcam-screen-http-proxy-jpxC1",
              help="Git branch to install from")
def update(branch: str) -> None:
    """Update face2face to the latest version from GitHub."""
    url = f"https://github.com/jtexp/face2face/archive/refs/heads/{branch}.zip"
    click.echo(f"Updating from branch '{branch}' ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", url])
    click.echo("Update complete.")


def main() -> None:
    """Entry point."""
    cli()


if __name__ == "__main__":
    main()
