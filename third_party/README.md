# third_party

Local reference checkouts of upstream repos. Not linked into the build; the
frontend files and runtime adapter point into these for reading only.

| Name | Local path | Remote |
|---|---|---|
| sysy-compiler-ref | `../../sysy-compiler` | https://github.com/BetterThanAny/sysy-compiler.git |
| mini-llm-engine-ref | `../../mini-llm-engine` | https://github.com/BetterThanAny/mini-llm-engine.git |

To refresh:

```bash
git -C ../sysy-compiler pull
git -C ../mini-llm-engine pull
```
