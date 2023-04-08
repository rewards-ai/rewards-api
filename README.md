### **Rewards API** 

<br>

**rewards API** is a `REST` API where creating experiments, integrating environments and building agents are just some sets of CRUD operations and API calls. Our API internally uses [**rewards-sdk**](https://github.com/rewards-ai/rewards-SDK). 

<br>

#### **How to run the project** 

First clone the project by running:
```bash
git clone https://github.com/rewards-ai/rewards-api.git
```

`rewards-api` runs on `fastapi`. So install fastapi by running: 

```bash
pip install fastapi
```

Now for installing additional dependencies just run:

```bash
pip install -r requirements.txt
```

Now go to the `rewards-api` directory and run:

```
uvicorn main:app --reload
```

This will open the the link `http://127.0.0.1:8000` and then go to `http://127.0.0.1:8000/docs/`. There you will find all the endpoints with it's instructions and curl commands after running each. 


TODO:

- [ ] Logging functionality
- [ ] Support for custom exceptions 
- [ ] Stopping the game using endpoint (Multithreading)
- [ ] Supporting streaming the game screen
- [ ] Integration with the frontend 