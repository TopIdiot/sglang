import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_manager import ServerStatus
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def _response(*, status_code=200, json_data=None, text=""):
    response = Mock(status_code=status_code, text=text)
    response.json.return_value = json_data
    return response


def _server_args(**overrides):
    values = dict(
        admin_api_key=None,
        api_key=None,
        debug_tensor_dump_input_file=None,
        disaggregation_mode="null",
        dp_size=1,
        skip_tokenizer_init=False,
        tp_size=1,
        welm_kv_mirror_pd_mode="deferred-last-prompt",
    )
    values.update(overrides)
    args = SimpleNamespace(**values)
    args.url = lambda: "http://127.0.0.1:30000"
    args.ssl_verify = lambda: False
    return args


def _mock_warmup_http(monkeypatch, post):
    tokenizer_manager = SimpleNamespace(server_status=ServerStatus.Starting)
    monkeypatch.setattr(
        http_server,
        "_global_state",
        SimpleNamespace(tokenizer_manager=tokenizer_manager),
    )
    monkeypatch.setattr(http_server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        http_server.requests,
        "get",
        Mock(
            return_value=_response(
                json_data={
                    "has_image_understanding": False,
                    "is_generation": True,
                }
            )
        ),
    )
    monkeypatch.setattr(http_server.requests, "post", post)
    return tokenizer_manager


def test_health_generate_uses_two_tokens_only_for_deferred(monkeypatch):
    async def run(mode):
        captured = []

        async def generate_request(req, _raw_request):
            captured.append(req)
            yield None

        tokenizer_manager = SimpleNamespace(
            gracefully_exit=False,
            server_status=ServerStatus.Up,
            is_generation=True,
            server_args=_server_args(welm_kv_mirror_pd_mode=mode),
            generate_request=generate_request,
            last_receive_tstamp=float("inf"),
            rid_to_state={},
        )
        monkeypatch.setattr(
            http_server,
            "_global_state",
            SimpleNamespace(tokenizer_manager=tokenizer_manager),
        )
        response = await http_server.health_generate(
            SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
        )
        assert response.status_code == 200
        assert len(captured) == 1
        return captured[0].input_ids

    assert asyncio.run(run("legacy")) == [0]
    assert asyncio.run(run("deferred-last-prompt")) == [0, 0]


def test_monolithic_deferred_server_warmup_runs_16k_request_and_flushes_cache(
    monkeypatch,
):
    post = Mock(side_effect=[_response(), _response(text="Cache flushed.")])
    tokenizer_manager = _mock_warmup_http(monkeypatch, post)
    monkeypatch.setattr(http_server.envs.SGLANG_WARMUP_TIMEOUT, "get", lambda: 37)

    assert http_server._execute_server_warmup(
        _server_args(api_key="api-key", admin_api_key="admin-key")
    )

    assert post.call_count == 2
    generate_call, flush_call = post.call_args_list
    assert generate_call.args[0] == "http://127.0.0.1:30000/generate"
    assert generate_call.kwargs["headers"] == {"Authorization": "Bearer api-key"}
    assert len(generate_call.kwargs["json"]["input_ids"]) == 16 * 1024
    assert generate_call.kwargs["json"]["sampling_params"]["ignore_eos"] is True
    assert flush_call.args[0] == "http://127.0.0.1:30000/flush_cache"
    assert flush_call.kwargs["headers"] == {"Authorization": "Bearer admin-key"}
    assert flush_call.kwargs["params"] == {"timeout": 37}
    assert tokenizer_manager.server_status is ServerStatus.Up
