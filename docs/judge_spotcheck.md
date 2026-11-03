# Judge spot-check

20 RAGAS-judged items sampled with seed 42 from the finalist configs (judge model `rag`). The human checks each verdict and ticks one box; the agent does not fill these in.

## 1. `q5b59a9349756` (config `cabfa4deefe8`)

**Question:** What happens if the path `/users/{user_id}` is defined before `/users/me` in the path operations list?

**Reference answer:** The path `/users/{user_id}` would incorrectly match `/users/me` and attempt to extract a `user_id` parameter with the value `"me"` instead of recognizing it as a fixed path.

**Generated answer:** The path for `/users/{user_id}` would match `/users/me`, interpreting `"me"` as the `user_id` parameter value.

**Retrieved contexts:**

1. ## Order matters { #order-matters } When creating *path operations*, you can find situations where you have a fixed path. Like `/users/me`, let's say that it's to get data about the current user. And then you can also have a path `/users/{user_id}` to get data about a specific user by some user ID. Because *path operations* are evaluated in order, you need to make sure that the path for `/users/me` is declared before the one for `/users/{user_id}`: {* ../../docs_src/path_params/tutorial003_py310...
2. ## Check it { #check-it } If you open the API docs, you can authenticate and specify which scopes you want to authorize. <img src="/img/tutorial/security/image11.png"> If you don't select any scope, you will be "authenticated", but when you try to access `/users/me/` or `/users/me/items/` you will get an error saying that you don't have enough permissions. You will still be able to access `/status/`. And if you select the scope `me` but not the scope `items`, you will be able to access `/users/m...
3. ## 0.39.0 (2019-09-29) * Allow path parameters to have default values (e.g. `None`) and discard them instead of raising an error. * This allows declaring a parameter like `user_id: str = None` that can be taken from a query parameter, but the same *path operation* can be included in a router with a path `/users/{user_id}`, in which case will be taken from the path and will be required. * PR [#464](https://github.com/tiangolo/fastapi/pull/464) by [@jonathanunderwood](https://github.com/jonathanun...
4. ## Declare scopes in *path operations* and dependencies { #declare-scopes-in-path-operations-and-dependencies } Now we declare that the *path operation* for `/users/me/items/` requires the scope `items`. For this, we import and use `Security` from `fastapi`. You can use `Security` to declare dependencies (just like `Depends`), but `Security` also receives a parameter `scopes` with a list of scopes (strings). In this case, we pass a dependency function `get_current_active_user` to `Security` (the...
5. ### Inactive user { #inactive-user } Now try with an inactive user, authenticate with: User: `alice` Password: `secret2` And try to use the operation `GET` with the path `/users/me`. You will get an "Inactive user" error, like: ```JSON { "detail": "Inactive user" } ```

| metric | judge score |
|---|---|
| faithfulness | 0.750 |
| answer_relevancy | 0.882 |
| context_precision | 0.833 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 2. `q99a1feb89b1a` (config `185f47fb8795`)

**Question:** What are two widely used Python packages that FastAPI allows you to integrate directly for handling password hashing and JWT token generation?

**Reference answer:** pwdlib and PyJWT

**Generated answer:** 1. **pwdlib** (for secure password hashing, including Argon2)
2. **PyJWT** (for JWT token generation)

**Retrieved contexts:**

1. ## Install `pwdlib` { #install-pwdlib } pwdlib is a great Python package to handle password hashes. It supports many secure hashing algorithms and utilities to work with them. The recommended algorithm is "Argon2". Add `pwdlib` with Argon2 to your project: <div class="termy"> ```console $ uv add "pwdlib[argon2]" ---> 100% ``` </div> /// tip With `pwdlib`, you could even configure it to be able to read passwords created by **Django**, a **Flask** security plug-in or many others. So, you would be ...
2. ## Recap { #recap } With what you have seen up to now, you can set up a secure **FastAPI** application using standards like OAuth2 and JWT. In almost any framework handling the security becomes a rather complex subject quite quickly. Many packages that simplify it a lot have to make many compromises with the data model, database, and available features. And some of these packages that simplify things too much actually have security flaws underneath. --- **FastAPI** doesn't make any compromise wi...
3. - ⚡ [**FastAPI**](https://fastapi.tiangolo.com) for the Python backend API. - 🧰 [SQLModel](https://sqlmodel.tiangolo.com) for the Python SQL database interactions (ORM). - 🔍 [Pydantic](https://pydantic.dev/docs/), used by FastAPI, for the data validation and settings management. - 💾 [PostgreSQL](https://www.postgresql.org) as the SQL database. - 🚀 [React](https://react.dev) for the frontend. - 💃 Using TypeScript, hooks, Vite, and other parts of a modern frontend stack. - 🎨 [Tailwind CSS](https:/...
4. ### Step 1: import `FastAPI` { #step-1-import-fastapi } {* ../../docs_src/first_steps/tutorial001_py310.py hl[1] *} `FastAPI` is a Python class that provides all the functionality for your API. /// note | Technical Details `FastAPI` is a class that inherits directly from `Starlette`. You can use all the [Starlette](https://starlette.dev/) functionality with `FastAPI` too. ///
5. ## **FastAPI** plug-ins { #fastapi-plug-ins } Integrations and "plug-ins" can be built using the **Dependency Injection** system. But in fact, there is actually **no need to create "plug-ins"**, as by using dependencies it's possible to declare an infinite number of integrations and interactions that become available to your *path operation functions*. And dependencies can be created in a very simple and intuitive way that allows you to just import the Python packages you need, and integrate the...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.843 |
| context_precision | 0.500 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 3. `q2f2a9821241b` (config `185f47fb8795`)

**Question:** What Python import is required to manually set a custom HTTP status code in a FastAPI endpoint?

**Reference answer:** You need to import `JSONResponse` (e.g., `from fastapi.responses import JSONResponse`).

**Generated answer:** from fastapi import status

**Retrieved contexts:**

1. # Status Codes You can import the `status` module from `fastapi`: ```python from fastapi import status ``` `status` is provided directly by Starlette. It contains a group of named constants (variables) with integer status codes. For example: * 200: `status.HTTP_200_OK` * 403: `status.HTTP_403_FORBIDDEN` * etc. It can be convenient to quickly access HTTP (and WebSocket) status codes in your app, using autocompletion for the name without having to memorize the integer status codes. Read more about...
2. ## Example ```python from fastapi import FastAPI, status app = FastAPI() @app.get("/items/", status_code=status.HTTP_418_IM_A_TEAPOT) def read_items(): return [{"name": "Plumbus"}, {"name": "Portal Gun"}] ``` ::: fastapi.status
3. ```Python from fastapi import Cookie, FastAPI, Header, Path, Query app = FastAPI() @app.get("/items/{item_id}") def main( item_id: int = Path(gt=0), query: str = Query(max_length=10), session: str = Cookie(min_length=3), x_trace: str = Header(title="Tracing header"), ): return {"message": "Hello World"} ``` To declare parameters as optional (not required), you can set a default value as always, for example using `None`: ```Python from typing import Union from fastapi import Cookie, FastAPI, Head...
4. ## Run the Server Program { #run-the-server-program } If you installed an ASGI server manually, you would normally need to pass an import string in a special format for it to import your FastAPI application: <div class="termy"> ```console $ uv run uvicorn main:app --host 0.0.0.0 --port 80 <span style="color: green;">INFO</span>: Uvicorn running on http://0.0.0.0:80 (Press CTRL+C to quit) ``` </div> /// note The command `uvicorn main:app` refers to: * `main`: the file `main.py` (the Python "modul...
5. ## Using `Depends` and others { #using-depends-and-others } In WebSocket endpoints you can import from `fastapi` and use: * `Depends` * `Security` * `Cookie` * `Header` * `Path` * `Query` They work the same way as for other FastAPI endpoints/*path operations*: {* ../../docs_src/websockets_/tutorial002_an_py310.py hl[68:69,82] *} /// note As this is a WebSocket it doesn't really make sense to raise an `HTTPException`, instead we raise a `WebSocketException`. You can use a closing code from the [v...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.804 |
| context_precision | 0.000 |
| context_recall | 0.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 4. `qd21f8893dfa4` (config `cabfa4deefe8`)

**Question:** How does FastAPI handle the execution of utility functions defined with `async def` compared to those defined with `def`?

**Reference answer:** Utility functions defined with `async def` require explicit `await` when called, and FastAPI does not automatically manage their execution in a threadpool. In contrast, functions defined with `def` are called directly as written in the code, without threadpool intervention.

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ### More technical details { #more-technical-details } You might have noticed that `await` can only be used inside of functions defined with `async def`. But at the same time, functions defined with `async def` have to be "awaited". So, functions with `async def` can only be called inside of functions defined with `async def` too. So, about the egg and the chicken, how do you call the first `async` function? If you are working with **FastAPI** you don't have to worry about that, because that "fi...
2. ### Other utility functions { #other-utility-functions } Any other utility function that you call directly can be created with normal `def` or `async def` and FastAPI won't affect the way you call it. This is in contrast to the functions that FastAPI calls for you: *path operation functions* and dependencies. If your utility function is a normal function with `def`, it will be called directly (as you write it in your code), not in a threadpool, if the function is created with `async def` then yo...
3. ```Python hl_lines="2" @app.get("/") def results(): results = some_library() return results ``` --- If your application (somehow) doesn't have to communicate with anything else and wait for it to respond, use `async def`, even if you don't need to use `await` inside. --- If you just don't know, use normal `def`. --- **Note**: You can mix `def` and `async def` in your *path operation functions* as much as you need and define each one using the best option for you. FastAPI will do the right thing ...
4. ### Path operation functions { #path-operation-functions } When you declare a *path operation function* with normal `def` instead of `async def`, it is run in an external threadpool that is then awaited, instead of being called directly (as it would block the server). If you are coming from another async framework that does not work in the way described above and you are used to defining trivial compute-only *path operation functions* with plain `def` for a tiny performance gain (about 100 nanos...
5. ## Create a task function { #create-a-task-function } Create a function to be run as the background task. It is just a standard function that can receive parameters. It can be an `async def` or normal `def` function, **FastAPI** will know how to handle it correctly. In this case, the task function will write to a file (simulating sending an email). And as the write operation doesn't use `async` and `await`, we define the function with normal `def`: {* ../../docs_src/background_tasks/tutorial001_...

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 0.500 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 5. `q71611b2094c8` (config `4443d42bc98a`)

**Question:** What are the two main scenarios where using `BackgroundTasks` is recommended over tools like Celery?

**Reference answer:** When you need to access variables and objects from the same FastAPI app or when performing small background tasks (such as sending an email notification).

**Generated answer:** If you need to access variables and objects from the same FastAPI app, or you need to perform small background tasks (like sending an email notification).

**Retrieved contexts:**

1. ## Caveat { #caveat } If you need to perform heavy background computation and you don't necessarily need it to be run by the same process (for example, you don't need to share memory, variables, etc), you might benefit from using other bigger tools like [Celery](https://docs.celeryq.dev). They tend to require more complex configurations, a message/job queue manager, like RabbitMQ or Redis, but they allow you to run background tasks in multiple processes, and especially, in multiple servers. But ...
2. ## Using `BackgroundTasks` { #using-backgroundtasks } First, import `BackgroundTasks` and define a parameter in your *path operation function* with a type declaration of `BackgroundTasks`: {* ../../docs_src/background_tasks/tutorial001_py310.py hl[1,13] *} **FastAPI** will create the object of type `BackgroundTasks` for you and pass it as that parameter.
3. ## Recap { #recap } Import and use `BackgroundTasks` with parameters in *path operation functions* and dependencies to add background tasks.
4. ## Dependency Injection { #dependency-injection } Using `BackgroundTasks` also works with the dependency injection system, you can declare a parameter of type `BackgroundTasks` at multiple levels: in a *path operation function*, in a dependency (dependable), in a sub-dependency, etc. **FastAPI** knows what to do in each case and how to reuse the same object, so that all the background tasks are merged together and are run in the background afterwards: {* ../../docs_src/background_tasks/tutorial0...
5. # Background Tasks - `BackgroundTasks` You can declare a parameter in a *path operation function* or dependency function with the type `BackgroundTasks`, and then you can use it to schedule the execution of background tasks after the response is sent. You can import it directly from `fastapi`: ```python from fastapi import BackgroundTasks ``` ::: fastapi.BackgroundTasks

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.642 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 6. `q5aa757c3b3ba` (config `4443d42bc98a`)

**Question:** What dependency injection mechanism in FastAPI allows handling security requirements for path operations?

**Reference answer:** The `Depends` mechanism at the Dependency Injection level.

**Generated answer:** `Depends` (context [3]).

**Retrieved contexts:**

1. ### Dependency Injection { #dependency-injection } FastAPI includes an extremely easy to use, but extremely powerful <dfn title='also known as "components", "resources", "services", "providers"'><strong>Dependency Injection</strong></dfn> system. * Even dependencies can have dependencies, creating a hierarchy or **"graph" of dependencies**. * All **automatically handled** by the framework. * All the dependencies can require data from requests and **augment the path operation** constraints and au...
2. ## What is "Dependency Injection" { #what-is-dependency-injection } **"Dependency Injection"** means, in programming, that there is a way for your code (in this case, your *path operation functions*) to declare things that it requires to work and use: "dependencies". And then, that system (in this case **FastAPI**) will take care of doing whatever is needed to provide your code with those needed dependencies ("inject" the dependencies). This is very useful when you need to: * Have shared logic (...
3. ## Other models { #other-models } You can now get the current user directly in the *path operation functions* and deal with the security mechanisms at the **Dependency Injection** level, using `Depends`. And you can use any model or data for the security requirements (in this case, a Pydantic model `User`). But you are not restricted to using some specific data model, class or type. Do you want to have an `id` and `email` and not have any `username` in your model? Sure. You can use these same to...
4. **In `FastAPI`** This allows setting the (mostly new) parameters (additionally to the already existing parameters): * `default_response_class`: updated to handle defaults in `APIRouter` and `include_router`. * `dependencies`: to include ✨ top-level dependencies ✨ that apply to the whole application. E.g. to add global authentication. * `callbacks`: OpenAPI callbacks that apply to all the *path operations*. * `deprecated`: to mark all the *path operations* as deprecated. 🤷 * `include_in_schema`: ...
5. ## Dependency Injection { #dependency-injection } Using `BackgroundTasks` also works with the dependency injection system, you can declare a parameter of type `BackgroundTasks` at multiple levels: in a *path operation function*, in a dependency (dependable), in a sub-dependency, etc. **FastAPI** knows what to do in each case and how to reuse the same object, so that all the background tasks are merged together and are run in the background afterwards: {* ../../docs_src/background_tasks/tutorial0...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.000 |
| context_precision | 0.806 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 7. `q32e359ce04c4` (config `4443d42bc98a`)

**Question:** What Python-specific example is provided in the documentation to illustrate adding different media types to a response?

**Reference answer:** The example uses `../../docs_src/additional_responses/tutorial002_py310.py` (lines 17:22,26) to demonstrate returning a JSON object or a PNG image via `FileResponse`.

**Generated answer:** The example in `tutorial002_py310.py` adds an `image/png` media type so a path operation can return either `application/json` or a PNG image (via `FileResponse`).

**Retrieved contexts:**

1. ## Additional media types for the main response { #additional-media-types-for-the-main-response } You can use this same `responses` parameter to add different media types for the same main response. For example, you can add an additional media type of `image/png`, declaring that your *path operation* can return a JSON object (with media type `application/json`) or a PNG image: {* ../../docs_src/additional_responses/tutorial002_py310.py hl[17:22,26] *} /// note Notice that you have to return the ...
2. ### `Response` { #response } The main `Response` class, all the other responses inherit from it. You can return it directly. It accepts the following parameters: * `content` - A `str` or `bytes`. * `status_code` - An `int` HTTP status code. * `headers` - A `dict` of strings. * `media_type` - A `str` giving the media type. E.g. `"text/html"`. FastAPI (actually Starlette) will automatically include a Content-Length header. It will also include a Content-Type header, based on the `media_type` and a...
3. ## 0.12.0 (2019-04-05) * Add additional `responses` parameter to *path operation decorators* to extend responses in OpenAPI (and API docs). * It also allows extending existing responses generated from `response_model`, declare other media types (like images), etc. * The new documentation is here: [Additional Responses](https://fastapi.tiangolo.com/advanced/additional-responses/). * `responses` can also be added to `.include_router()`, the updated docs are here: [Bigger Applications](https://fast...
4. ### `FileResponse` { #fileresponse } Asynchronously streams a file as the response. Takes a different set of arguments to instantiate than the other response types: * `path` - The file path to the file to stream. * `headers` - Any custom headers to include, as a dictionary. * `media_type` - A string giving the media type. If unset, the filename or path will be used to infer a media type. * `filename` - If set, this will be included in the response `Content-Disposition`. File responses will inclu...
5. ## More information about OpenAPI responses { #more-information-about-openapi-responses } To see what exactly you can include in the responses, you can check these sections in the OpenAPI specification: * [OpenAPI Responses Object](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#responses-object), it includes the `Response Object`. * [OpenAPI Response Object](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.1.0.md#response-object), you can include anythi...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.764 |
| context_precision | 0.833 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 8. `qc55dca7b033b` (config `185f47fb8795`)

**Question:** What FastAPI version introduced the `separate_input_output_schemas` parameter?

**Reference answer:** FastAPI `0.102.0`

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ## Do not Separate Schemas { #do-not-separate-schemas } Now, there are some cases where you might want to have the **same schema for input and output**. Probably the main use case for this is if you already have some autogenerated client code/SDKs and you don't want to update all the autogenerated client code/SDKs yet, you probably will want to do it at some point, but maybe not right now. In that case, you can disable this feature in **FastAPI**, with the parameter `separate_input_output_schema...
2. ### Docs * 📝 Add note to docs about Separate Input and Output Schemas with FastAPI version. PR [#10150](https://github.com/tiangolo/fastapi/pull/10150) by [@tiangolo](https://github.com/tiangolo).
3. ### Features * ✨ Add support for disabling the separation of input and output JSON Schemas in OpenAPI with Pydantic v2 with `separate_input_output_schemas=False`. PR [#10145](https://github.com/tiangolo/fastapi/pull/10145) by [@tiangolo](https://github.com/tiangolo). * New docs [Separate OpenAPI Schemas for Input and Output or Not](https://fastapi.tiangolo.com/how-to/separate-openapi-schemas/). * This PR also includes a new setup (internal tools) for generating screenshots for the docs.
4. ### Fixes * 🐛 Fix OpenAPI schema support for computed fields when using `separate_input_output_schemas=False`. PR [#13207](https://github.com/fastapi/fastapi/pull/13207) by [@vgrafe](https://github.com/vgrafe).
5. ## Version { #version } This is available since FastAPI version `0.67.0`. 🔖

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 9. `qcf155620b3bf` (config `cabfa4deefe8`)

**Question:** What happens if you set `convert_underscores` to `False` in the `Header` class and why should you be cautious about this choice?

**Reference answer:** The `Header` class will no longer automatically convert underscores (`_`) to hyphens (`-`) in header names. This is risky because some HTTP proxies and servers reject headers containing underscores, potentially causing compatibility issues.

**Generated answer:** If you set `convert_underscores` to `False` in the `Header` class, the automatic conversion of underscores (`_`) to hyphens (`-`) in header parameter names is disabled. This means headers like `user_agent` will be expected as-is in HTTP requests, not as `user-agent`.

You should be cautious about this choice because **some HTTP proxies and servers disallow the usage of headers with underscores** (as mentioned in the warnings in [1] and [2]).

**Retrieved contexts:**

1. ## Automatic conversion { #automatic-conversion } `Header` has a little extra functionality on top of what `Path`, `Query` and `Cookie` provide. Most of the standard headers are separated by a "hyphen" character, also known as the "minus symbol" (`-`). But a variable like `user-agent` is invalid in Python. So, by default, `Header` will convert the parameter names characters from underscore (`_`) to hyphen (`-`) to extract and document the headers. Also, HTTP headers are case-insensitive, so, you...
2. ## Disable Convert Underscores { #disable-convert-underscores } The same way as with regular header parameters, when you have underscore characters in the parameter names, they are **automatically converted to hyphens**. For example, if you have a header parameter `save_data` in the code, the expected HTTP header will be `save-data`, and it will show up like that in the docs. If for some reason you need to disable this automatic conversion, you can do it as well for Pydantic models for header pa...
3. ## 0.152.0 (2026-10-20) Validation errors for header parameter models now report the HTTP header name, such as `int-val`, instead of the Python field name, `int_val`. Explicit aliases, `Header(convert_underscores=False)`, and Pydantic models with `loc_by_alias=False` keep their existing behavior.
4. ### Fixes * 🐛 Fix `convert_underscores=False` for header Pydantic models. PR [#13515](https://github.com/fastapi/fastapi/pull/13515) by [@tiangolo](https://github.com/tiangolo).
5. Values such as `0`, `false`, `off`, and `no` (including case variations) are converted to `False`. If `short` is omitted, **FastAPI** uses its default value, which is `False` in this example. An empty value (`?short=`) or an unrecognized value (for example, `?short=maybe`) produces a `422` validation error.

| metric | judge score |
|---|---|
| faithfulness | 0.800 |
| answer_relevancy | 0.938 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 10. `q8a716345f84b` (config `185f47fb8795`)

**Question:** What happens if you return a database model after setting cookies in a temporary response object?

**Reference answer:** FastAPI will use the temporary response to extract the cookies (and headers/status code) and include them in the final response, which will contain the filtered/serialized model via the `response_model`.

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ## Use a `Response` parameter { #use-a-response-parameter } You can declare a parameter of type `Response` in your *path operation function*. And then you can set cookies in that *temporary* response object. {* ../../docs_src/response_cookies/tutorial002_py310.py hl[1, 8:9] *} And then you can return any object you need, as you normally would (a `dict`, a database model, etc). And if you declared a `response_model`, it will still be used to filter and convert the object you returned. **FastAPI**...
2. ## Use a `Response` parameter { #use-a-response-parameter } You can declare a parameter of type `Response` in your *path operation function* (as you can do for cookies). And then you can set headers in that *temporary* response object. {* ../../docs_src/response_headers/tutorial002_py310.py hl[1, 7:8] *} And then you can return any object you need, as you normally would (a `dict`, a database model, etc). And if you declared a `response_model`, it will still be used to filter and convert the obje...
3. ## Use a `Response` parameter { #use-a-response-parameter } You can declare a parameter of type `Response` in your *path operation function* (as you can do for cookies and headers). And then you can set the `status_code` in that *temporary* response object. {* ../../docs_src/response_change_status_code/tutorial001_py310.py hl[1,9,12] *} And then you can return any object you need, as you normally would (a `dict`, a database model, etc). And if you declared a `response_model`, it will still be us...
4. ## Return a `Response` directly { #return-a-response-directly } You can also create cookies when returning a `Response` directly in your code. To do that, you can create a response as described in [Return a Response Directly](response-directly.md). Then set Cookies in it, and then return it: {* ../../docs_src/response_cookies/tutorial001_py310.py hl[10:12] *} /// tip Keep in mind that if you return a response directly instead of using the `Response` parameter, FastAPI will return it directly. So...
5. ## `response_model` Parameter { #response-model-parameter } There are some cases where you need or want to return some data that is not exactly what the type declares. For example, you could want to **return a dictionary** or a database object, but **declare it as a Pydantic model**. This way the Pydantic model would do all the data documentation, validation, etc. for the object that you returned (e.g. a dictionary or database object). If you added the return type annotation, tools and editors w...

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 0.950 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 11. `q78dde312af6d` (config `cabfa4deefe8`)

**Question:** What command should you use to install the FastAPI skill for AI coding agents if you have `uvx` configured as an alias for `uv tool run`?

**Reference answer:** uvx library-skills

**Generated answer:** ```bash
uvx library-skills
```

**Retrieved contexts:**

1. ## AI Agent Skills { #ai-agent-skills } FastAPI includes an official skill for AI coding agents. It is bundled with the package, so its guidance stays aligned with the version of FastAPI installed in your project and updates when you update FastAPI. After installing FastAPI in your project, you can install the skill with <a href="https://library-skills.io">Library Skills</a>: ```bash uvx library-skills ``` /// note `uvx` is an alias for `uv tool run`. It runs Library Skills in a temporary, isola...
2. ## Install FastAPI { #install-fastapi } The first step is to set up your project and add FastAPI. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/), then create a project and add FastAPI: <div class="termy"> ```console $ uv init awesome-project --bare $ cd awesome-project $ uv add "fastapi[standard]" ---> 100% ``` </div> `uv add` creates the project's virtual environment in `.venv`, adds FastAPI to `pyproject.toml`, and creates `uv.lock` so the same package versions can be ...
3. ## Install the Server Program { #install-the-server-program } When you install FastAPI, it comes with a production server, Uvicorn, and you can start it with the `fastapi run` command. But you can also install an ASGI server manually. Add the server application to your project. For example, to install Uvicorn: <div class="termy"> ```console $ uv add "uvicorn[standard]" ---> 100% ``` </div> A similar process would apply to any other ASGI server program. /// tip By adding the `standard`, Uvicorn w...
4. ### `fastapi dev` with path or with `--entrypoint` CLI option { #fastapi-dev-with-path-or-with-entrypoint-cli-option } You can also pass the file path to the `fastapi dev` command, and it will guess the FastAPI app object to use: ```console $ uv run fastapi dev main.py ``` Or, you can also pass the `--entrypoint` option to the `fastapi dev` command: ```console $ uv run fastapi dev --entrypoint main:app ``` But you would have to remember to pass the correct path\entrypoint every time you call the...
5. ### `fastapi dev` with path or with `--entrypoint` CLI option { #fastapi-dev-with-path-or-with-entrypoint-cli-option } You can also pass the file path to the `fastapi dev` command, and it will guess the FastAPI app object to use: ```console $ uv run fastapi dev main.py ``` Or, you can also pass the `--entrypoint` option to the `fastapi dev` command: ```console $ uv run fastapi dev --entrypoint main:app ``` But you would have to remember to pass the correct path\entrypoint every time you call the...

| metric | judge score |
|---|---|
| faithfulness | 0.750 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 12. `q991092ed0ffc` (config `ceb37f9c707e`)

**Question:** What happens if two form models declare the same field name when combined?

**Reference answer:** Each model validates that field's value independently, so the field must have compatible types and constraints across models to ensure the OpenAPI schema can describe them consistently.

**Generated answer:** Each model validates that field's value independently. Use compatible types and constraints for shared fields so the OpenAPI schema can describe them consistently.

**Retrieved contexts:**

1. ## Combine Form Models { #combine-form-models } You can combine multiple form models with individual `Form` parameters. {* ../../docs_src/request_form_models/tutorial003_an_py310.py hl[9:14,19:21] *} The client sends `name`, `tags`, and `token` as form fields. Each model validates the form fields independently. Each model's `extra` configuration applies to all form fields, including fields and files declared in other parameters or dependencies. With `extra="forbid"`, any field outside that model...
2. ## Combine Unions of Form Models { #combine-unions-of-form-models } You can combine unions with other form models, individual `Form` parameters, and `File` parameters. Each union is validated independently. {* ../../docs_src/request_form_models/tutorial007_an_py310.py hl[31:35] *} Send all fields in the same form. For example, send `contact_type=person`, `name=Rick`, `notification_type=email`, `email=rick@example.com`, and `token=secret`. Omit the notification fields to leave `notification` as `...
3. ## 0.160.0 (2026-10-22) Query, header, cookie, and form parameter models now apply their `extra` configuration to all submitted fields in the same parameter location, consistent with Pydantic behavior. This includes fields declared in other models, individual parameters, or dependencies, which were previously excluded. With `extra="forbid"`, fields outside the model are rejected even if another parameter declares them. With `extra="allow"`, those fields are included as model extras. With the def...
4. ## Combine Parameter Models { #combine-parameter-models } You can declare multiple query parameter models, and combine them with individual query parameters or dependencies. {* ../../docs_src/query_param_models/tutorial003_an_py310.py hl[9:15,20:22] *} The request parameters stay at the same level. ```http https://example.com/items/?limit=5&offset=10&tags=red&tags=blue&search=shirt ``` There are no `pagination` or `filters` keys in the query string. Each model validates the query parameters inde...
5. # Form Models { #form-models } You can use **Pydantic models** to declare **form fields** in FastAPI. /// note To use forms, first install [`python-multipart`](https://github.com/Kludex/python-multipart). Add it to your project: ```console $ uv add python-multipart ``` /// /// note This is supported since FastAPI version `0.113.0`. 🤓 ///

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.624 |
| context_precision | 0.750 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 13. `qd21f8893dfa4` (config `47f27b36124e`)

**Question:** How does FastAPI handle the execution of utility functions defined with `async def` compared to those defined with `def`?

**Reference answer:** Utility functions defined with `async def` require explicit `await` when called, and FastAPI does not automatically manage their execution in a threadpool. In contrast, functions defined with `def` are called directly as written in the code, without threadpool intervention.

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ### More technical details { #more-technical-details } You might have noticed that `await` can only be used inside of functions defined with `async def`. But at the same time, functions defined with `async def` have to be "awaited". So, functions with `async def` can only be called inside of functions defined with `async def` too. So, about the egg and the chicken, how do you call the first `async` function? If you are working with **FastAPI** you don't have to worry about that, because that "fi...
2. ### Other utility functions { #other-utility-functions } Any other utility function that you call directly can be created with normal `def` or `async def` and FastAPI won't affect the way you call it. This is in contrast to the functions that FastAPI calls for you: *path operation functions* and dependencies. If your utility function is a normal function with `def`, it will be called directly (as you write it in your code), not in a threadpool, if the function is created with `async def` then yo...
3. ```Python hl_lines="2" @app.get("/") def results(): results = some_library() return results ``` --- If your application (somehow) doesn't have to communicate with anything else and wait for it to respond, use `async def`, even if you don't need to use `await` inside. --- If you just don't know, use normal `def`. --- **Note**: You can mix `def` and `async def` in your *path operation functions* as much as you need and define each one using the best option for you. FastAPI will do the right thing ...
4. ### Path operation functions { #path-operation-functions } When you declare a *path operation function* with normal `def` instead of `async def`, it is run in an external threadpool that is then awaited, instead of being called directly (as it would block the server). If you are coming from another async framework that does not work in the way described above and you are used to defining trivial compute-only *path operation functions* with plain `def` for a tiny performance gain (about 100 nanos...
5. ## Create a task function { #create-a-task-function } Create a function to be run as the background task. It is just a standard function that can receive parameters. It can be an `async def` or normal `def` function, **FastAPI** will know how to handle it correctly. In this case, the task function will write to a file (simulating sending an email). And as the write operation doesn't use `async` and `await`, we define the function with normal `def`: {* ../../docs_src/background_tasks/tutorial001_...

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 0.500 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 14. `q75ae18ca9d92` (config `185f47fb8795`)

**Question:** What HTTP status code does `RedirectResponse` use when no status code is explicitly provided?

**Reference answer:** 307

**Generated answer:** 307.

**Retrieved contexts:**

1. ### `RedirectResponse` { #redirectresponse } Returns an HTTP redirect. Uses a 307 status code (Temporary Redirect) by default. You can return a `RedirectResponse` directly: {* ../../docs_src/custom_response/tutorial006_py310.py hl[2,9] *} --- Or you can use it in the `response_class` parameter: {* ../../docs_src/custom_response/tutorial006b_py310.py hl[2,7,9] *} If you do that, then you can return the URL directly from your *path operation* function. In this case, the `status_code` used will be ...
2. ### Fixes * 🐛 Fix `RuntimeError` raised when `HTTPException` has a status code with no content. PR [#5365](https://github.com/tiangolo/fastapi/pull/5365) by [@iudeen](https://github.com/iudeen). * 🐛 Fix empty response body when default `status_code` is empty but the `Response` parameter with `response.status_code` is set. PR [#5360](https://github.com/tiangolo/fastapi/pull/5360) by [@tmeckel](https://github.com/tmeckel).
3. ## Use case { #use-case } For example, imagine that you want to return an HTTP status code of "OK" `200` by default. But if the data didn't exist, you want to create it, and return an HTTP status code of "CREATED" `201`. But you still want to be able to filter and convert the data you return with a `response_model`. For those cases, you can use a `Response` parameter.
4. ### Return the error { #return-the-error } After detecting that the credentials are incorrect, return an `HTTPException` with a status code 401 (the same returned when no credentials are provided) and add the header `WWW-Authenticate` to make the browser show the login prompt again: {* ../../docs_src/security/tutorial007_an_py310.py hl[26:30] *}
5. ## Additional status codes { #additional-status-codes_1 } If you want to return additional status codes apart from the main one, you can do that by returning a `Response` directly, like a `JSONResponse`, and set the additional status code directly. For example, let's say that you want to have a *path operation* that allows updating items, and returns HTTP status codes of 200 "OK" when successful. But you also want it to accept new items. And when the items didn't exist before, it creates them, a...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 15. `q10b7fc563c09` (config `cabfa4deefe8`)

**Question:** What are the two primary libraries that SQLModel is built upon, and who created it?

**Reference answer:** SQLModel is built on top of SQLAlchemy and Pydantic, and it was created by the same author as FastAPI.

**Generated answer:** SQLModel is built on **SQLAlchemy** and **Pydantic**. It was created by the same author as **FastAPI**.

**Retrieved contexts:**

1. # SQL (Relational) Databases { #sql-relational-databases } **FastAPI** doesn't require you to use a SQL (relational) database. But you can use **any database** that you want. Here we'll see an example using [SQLModel](https://sqlmodel.tiangolo.com/). **SQLModel** is built on top of [SQLAlchemy](https://www.sqlalchemy.org/) and Pydantic. It was made by the same author of **FastAPI** to be the perfect match for FastAPI applications that need to use **SQL databases**. /// tip You could use any othe...
2. **Note:** We use `int | None` for the primary key field so that in Python code we can *create an object without an `id`* (`id=None`), assuming the database will *generate it when saving*. SQLModel understands that the database will provide the `id` and *defines the column as a non-null `INTEGER`* in the database schema. See [SQLModel docs on primary keys](https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/#primary-key-id) for details. * `Field(index=True)` tells SQLModel that it should c...
3. * PostgreSQL * MySQL * SQLite * Oracle * Microsoft SQL Server, etc. In this example, we'll use **SQLite**, because it uses a single file and Python has integrated support. So, you can copy this example and run it as is. Later, for your production application, you might want to use a database server like **PostgreSQL**. /// tip There is an official project generator with **FastAPI** and **PostgreSQL** including a frontend and more tools: [https://github.com/fastapi/full-stack-fastapi-template](ht...
4. ## Install `SQLModel` { #install-sqlmodel } Add `sqlmodel` to your project: <div class="termy"> ```console $ uv add sqlmodel ---> 100% ``` </div>
5. ### Create Models { #create-models } Import `SQLModel` and create a database model: {* ../../docs_src/sql_databases/tutorial001_an_py310.py ln[1:11] hl[7:11] *} The `Hero` class is very similar to a Pydantic model (in fact, underneath, it actually *is a Pydantic model*). There are a few differences: * `table=True` tells SQLModel that this is a *table model*, it should represent a **table** in the SQL database, it's not just a *data model* (as would be any other regular Pydantic class). * `Field(...

| metric | judge score |
|---|---|
| faithfulness | 0.833 |
| answer_relevancy | 0.848 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 16. `q400c9c6ba0de` (config `47f27b36124e`)

**Question:** How can a Pydantic model be utilized to generate a JSON Schema for a non-JSON request format like YAML in FastAPI?

**Reference answer:** By manually creating the JSON Schema from the Pydantic model and then parsing the YAML request body as bytes in code, validating it with the same Pydantic model afterward.

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ### Custom OpenAPI content type { #custom-openapi-content-type } Using this same trick, you could use a Pydantic model to define the JSON Schema that is then included in the custom OpenAPI schema section for the *path operation*. And you could do this even if the data type in the request is not JSON. For example, in this application we don't use FastAPI's integrated functionality to extract the JSON Schema from Pydantic models nor the automatic validation for JSON. In fact, we are declaring the ...
2. **FastAPI** will take the Pydantic model from there, generate the JSON Schema, and put it in the correct place. The correct place is: * In the key `content`, that has as value another JSON object (`dict`) that contains: * A key with the media type, e.g. `application/json`, that contains as value another JSON object, that contains: * A key `schema`, that has as the value the JSON Schema from the model, here's the correct place. * **FastAPI** adds a reference here to the global JSON Schemas in ano...
3. ## Extra JSON Schema data in Pydantic models { #extra-json-schema-data-in-pydantic-models } You can declare `examples` for a Pydantic model that will be added to the generated JSON Schema. {* ../../docs_src/schema_extra_example/tutorial001_py310.py hl[13:24] *} That extra info will be added as-is to the output **JSON Schema** for that model, and it will be used in the API docs. You can use the attribute `model_config` that takes a `dict` as described in [Pydantic's docs: Configuration](https://p...
4. ### `orjson` or Response Model { #orjson-or-response-model } If what you are looking for is performance, you are probably better off using a [Response Model](../tutorial/response-model.md) than an `orjson` response. With a response model, FastAPI will use Pydantic to serialize the data to JSON, without using intermediate steps, like converting it with `jsonable_encoder`, which would happen in any other case. And under the hood, Pydantic uses the same underlying Rust mechanisms as `orjson` to ser...
5. ### Pydantic and FastAPI `examples` { #pydantic-and-fastapi-examples } When you add `examples` inside a Pydantic model, using `schema_extra` or `Field(examples=["something"])` that example is added to the **JSON Schema** for that Pydantic model. And that **JSON Schema** of the Pydantic model is included in the **OpenAPI** of your API, and then it's used in the docs UI. In versions of FastAPI before 0.99.0 (0.99.0 and above use the newer OpenAPI 3.1.0) when you used `example` or `examples` with a...

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 17. `q400c9c6ba0de` (config `185f47fb8795`)

**Question:** How can a Pydantic model be utilized to generate a JSON Schema for a non-JSON request format like YAML in FastAPI?

**Reference answer:** By manually creating the JSON Schema from the Pydantic model and then parsing the YAML request body as bytes in code, validating it with the same Pydantic model afterward.

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ### Custom OpenAPI content type { #custom-openapi-content-type } Using this same trick, you could use a Pydantic model to define the JSON Schema that is then included in the custom OpenAPI schema section for the *path operation*. And you could do this even if the data type in the request is not JSON. For example, in this application we don't use FastAPI's integrated functionality to extract the JSON Schema from Pydantic models nor the automatic validation for JSON. In fact, we are declaring the ...
2. **FastAPI** will take the Pydantic model from there, generate the JSON Schema, and put it in the correct place. The correct place is: * In the key `content`, that has as value another JSON object (`dict`) that contains: * A key with the media type, e.g. `application/json`, that contains as value another JSON object, that contains: * A key `schema`, that has as the value the JSON Schema from the model, here's the correct place. * **FastAPI** adds a reference here to the global JSON Schemas in ano...
3. ## Additional Response with `model` { #additional-response-with-model } You can pass to your *path operation decorators* a parameter `responses`. It receives a `dict`: the keys are status codes for each response (like `200`), and the values are other `dict`s with the information for each of them. Each of those response `dict`s can have a key `model`, containing a Pydantic model, just like `response_model`. **FastAPI** will take that model, generate its JSON Schema and include it in the correct p...
4. ## Extra JSON Schema data in Pydantic models { #extra-json-schema-data-in-pydantic-models } You can declare `examples` for a Pydantic model that will be added to the generated JSON Schema. {* ../../docs_src/schema_extra_example/tutorial001_py310.py hl[13:24] *} That extra info will be added as-is to the output **JSON Schema** for that model, and it will be used in the API docs. You can use the attribute `model_config` that takes a `dict` as described in [Pydantic's docs: Configuration](https://p...
5. ### `orjson` or Response Model { #orjson-or-response-model } If what you are looking for is performance, you are probably better off using a [Response Model](../tutorial/response-model.md) than an `orjson` response. With a response model, FastAPI will use Pydantic to serialize the data to JSON, without using intermediate steps, like converting it with `jsonable_encoder`, which would happen in any other case. And under the hood, Pydantic uses the same underlying Rust mechanisms as `orjson` to ser...

| metric | judge score |
|---|---|
| faithfulness | 0.000 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 18. `q3e19c9e1e763` (config `185f47fb8795`)

**Question:** What does a change in the middle number of a version (e.g., 0.2.3 → 0.3.3) indicate in FastAPI’s versioning scheme?

**Reference answer:** Breaking changes and new features are added in 'MINOR' versions (the middle number).

**Generated answer:** Breaking changes and new features.

**Retrieved contexts:**

1. ## About versions { #about-versions } Following the Semantic Versioning conventions, any version below `1.0.0` could potentially add breaking changes. FastAPI also follows the convention that any "PATCH" version change is for bug fixes and non-breaking changes. /// tip The "PATCH" is the last number, for example, in `0.2.3`, the PATCH version is `3`. /// So, you should be able to pin to a version like: ```txt fastapi>=0.45.0,<0.46.0 ``` Breaking changes and new features are added in "MINOR" vers...
2. # About FastAPI versions { #about-fastapi-versions } **FastAPI** is already being used in production in many applications and systems. And the test coverage is kept at 100%. But its development is still moving quickly. New features are added frequently, bugs are fixed regularly, and the code is still continuously improving. That's why the current versions are still `0.x.x`, this reflects that each version could potentially have breaking changes. This follows the [Semantic Versioning](https://sem...
3. ## Version { #version } This is available since FastAPI version `0.67.0`. 🔖
4. ### Features * ✨ Add docs and tests for Python 3.9 and Python 3.10. PR [#3712](https://github.com/tiangolo/fastapi/pull/3712) by [@tiangolo](https://github.com/tiangolo). * You can start with [Python Types Intro](https://fastapi.tiangolo.com/python-types/), it explains what changes between different Python versions, in Python 3.9 and in Python 3.10. * All the FastAPI docs are updated. Each code example in the docs that could use different syntax in Python 3.9 or Python 3.10 now has all the alter...
5. ## Pin your `fastapi` version { #pin-your-fastapi-version } The first thing you should do is to "pin" the version of **FastAPI** you are using to the specific latest version that you know works correctly for your application. For example, let's say you are using version `0.112.0` in your app. If you use a `requirements.txt` file you could specify the version with: ```txt fastapi[standard]==0.112.0 ``` that would mean that you would use exactly the version `0.112.0`. Or you could also pin it with...

| metric | judge score |
|---|---|
| faithfulness | 0.500 |
| answer_relevancy | 0.000 |
| context_precision | 1.000 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 19. `q7aedcc4df4ab` (config `185f47fb8795`)

**Question:** What are the key considerations mentioned in the documentation that should be kept in mind when configuring a deployment?

**Reference answer:** Security (HTTPS), running on startup, restarts, replication (number of processes), memory, and previous steps before starting are the main concepts to consider when deploying an application.

**Generated answer:** 1. Security - HTTPS
2. Running on startup
3. Restarts
4. Replication (the number of processes running)
5. Memory
6. Previous steps before starting

**Retrieved contexts:**

1. ## Recap { #recap } You have been reading here some of the main concepts that you would probably need to keep in mind when deciding how to deploy your application: * Security - HTTPS * Running on startup * Restarts * Replication (the number of processes running) * Memory * Previous steps before starting Understanding these ideas and how to apply them should give you the intuition necessary to take any decisions when configuring and tweaking your deployments. 🤓 In the next sections, I'll give you...
2. ## Deployment Strategies { #deployment-strategies } There are several ways to do it depending on your specific use case and the tools that you use. You could **deploy a server** yourself using a combination of tools, you could use a **cloud service** that does part of the work for you, or other possible options. For example, we, the team behind FastAPI, built [**FastAPI Cloud**](https://fastapicloud.com), to make deploying FastAPI apps to the cloud as streamlined as possible, with the same devel...
3. * the deployment * to deploy * the SDK * the software development kit * the `APIRouter` * the `requirements.txt` * the Bearer Token * the breaking change * the bug * the button * the callable * the code * the commit * the context manager * the coroutine * the database session * the disk * the domain * the engine * the fake X * the HTTP GET method * the item * the library * the lifespan * the lock * the middleware * the mobile application * the module * the mounting * the network * the origin * t...
4. ## What Does Deployment Mean { #what-does-deployment-mean } To **deploy** an application means to perform the necessary steps to make it **available to the users**. For a **web API**, it normally involves putting it in a **remote machine**, with a **server program** that provides good performance, stability, etc, so that your **users** can **access** the application efficiently and without interruptions or problems. This is in contrast to the **development** stages, where you are constantly chan...
5. # Deployments Concepts { #deployments-concepts } When deploying a **FastAPI** application, or actually, any type of web API, there are several concepts that you probably care about, and using them you can find the **most appropriate** way to **deploy your application**. Some of the important concepts are: * Security - HTTPS * Running on startup * Restarts * Replication (the number of processes running) * Memory * Previous steps before starting We'll see how they would affect **deployments**. In ...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.700 |
| context_precision | 0.700 |
| context_recall | 1.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

## 20. `q2e5fe84dd4fe` (config `4443d42bc98a`)

**Question:** How much additional RAM would be required if an API with a 1.5 GB model runs with 6 concurrent processes?

**Reference answer:** The API would require an additional 8 GB of RAM (6 processes × 1.5 GB each).

**Generated answer:** The provided context does not contain the answer.

**Retrieved contexts:**

1. ### Server Memory { #server-memory } For example, if your code loads a Machine Learning model with **1 GB in size**, when you run one process with your API, it will consume at least 1 GB of RAM. And if you start **4 processes** (4 workers), each will consume 1 GB of RAM. So in total, your API will consume **4 GB of RAM**. And if your remote server or virtual machine only has 3 GB of RAM, trying to load more than 4 GB of RAM will cause problems. 🚨
2. On the other hand, if you have 2 servers and you are using **100% of their CPU and RAM**, at some point one process will ask for more memory, and the server will have to use the disk as "memory" (which can be thousands of times slower), or even **crash**. Or one process might need to do some computation and would have to wait until the CPU is free again. In this case, it would be better to get **one extra server** and run some processes on it so that they all have **enough RAM and CPU time**. Th...
3. ## Replication - Processes and Memory { #replication-processes-and-memory } With a FastAPI application, using a server program like the `fastapi` command that runs Uvicorn, running it once in **one process** can serve multiple clients concurrently. But in many cases, you will want to run several worker processes at the same time.
4. ### Multiple Processes - An Example { #multiple-processes-an-example } In this example, there's a **Manager Process** that starts and controls two **Worker Processes**. This Manager Process would probably be the one listening on the **port** in the IP. And it would transmit all the communication to the worker processes. Those worker processes would be the ones running your application, they would perform the main computations to receive a **request** and return a **response**, and they would loa...
5. ## Memory { #memory } If you run **a single process per container** you will have a more or less well-defined, stable, and limited amount of memory consumed by each of those containers (more than one if they are replicated). And then you can set those same memory limits and requirements in your configurations for your container management system (for example in **Kubernetes**). That way it will be able to **replicate the containers** in the **available machines** taking into account the amount o...

| metric | judge score |
|---|---|
| faithfulness | 1.000 |
| answer_relevancy | 0.000 |
| context_precision | 0.000 |
| context_recall | 0.000 |

- [ ] Judge verdicts look right
- [ ] Judge verdicts look wrong (note which metric and why):

