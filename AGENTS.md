north star:

```
edit perf_takehome.py
python perf_takehome.py Tests.test_kernel_cycles # goal is under 1487 cycles
python reports/generate_report.py # REPORT.md
```

if considering improvements:
- consider REPORT.md
- maintain IDEAS.md (can be used as a brainstorm and a to do list)
- consider performance improvements (this is our main goal)
- consider improvements to the codebase or observability that will make performance improvements easier to identify or implement.

after making changes:
- run the tests
- update PROGRESS.md with a summary and the new cycle count
- if theres a regression then consider reverting the code change but keep the progress information
- make other docs and link to them from PROGRESS.md only if there are useful learnings that are likely to help witb future iterations
- remove things from IDEAS.md when done

