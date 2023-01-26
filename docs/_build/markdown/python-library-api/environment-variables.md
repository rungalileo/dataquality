---
description: Environment Variables to modify the behavior of Galileo
---

# Environment Variables

### GALILEO\_DISABLED

You can set the environment variable `GALILEO_DISABLED="TRUE"` to stop all Galileo functionality. This can be useful if you'd like to keep your Galileo logging code in your script, but train your model without actually logging to Galileo

### GALILEO\_VERBOSE

You can set the environment variable `GALILEO_VERBOSE="TRUE"` to force verbose logging in Galileo. This comes in handy when debugging certain issues in your model logging (like logging duplicate outputs).&#x20;

### GALILEO\_CONSOLE\_URL

You can set the environment variable GALILEO\_CONSOLE\_URL`="console.cloud.rungalileo.io"` (or your enterprise Galileo console url) to point to your custom Galileo deployment .

### GALILEO\_USERNAME

You can set the environment variable `GALILEO_USERNAME="me@mydomain.com"` to skip the prompting of your email during login. This variable should be set to your **email** for Galileo.

### GALILEO\_PASSWORD

You can set the environment variable `GALILEO_PASSWORD="my_aw3s0m3_p@ssword"` to skip the prompting of your password during login



