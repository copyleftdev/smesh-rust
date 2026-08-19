# SMESH film — narration script

Voice: Brian (ElevenLabs nPczCjzI2devNBz1zQrb)  ·  19 segments

## s01_cold_open  (roots)

Under every forest, there is a second network. It is older than ours. Trees use it to warn each other — about drought, about insects, about disease. No tree is in charge of it. And yet, somehow, the whole forest knows.

## s02_mechanism  (forest)

It runs on three simple rules. A tree in trouble releases a signal into the network. That signal fades as it travels, and fades as time passes — so old news disappears on its own. But when a second tree senses the same threat, and releases the same signal, the two reinforce each other. The message gets stronger. Nothing coordinates any of this. There is no root server. There is no forest manager. Coordination is not something the forest does. It is something the forest grows.

## s03_reveal  (reveal)

We built a protocol that works the same way. It is called SMESH. Software agents that coordinate the way a forest does — by releasing signals that fade, and trusting the ones that other agents independently confirm.

## s04_problem  (problem)

Here is why that matters. Almost every distributed system we build today has something sitting in the middle. A message broker. An orchestrator. A coordinator. It is the thing that knows everything, and tells everyone else what to do. It is also the thing you pay for. The thing you scale. The thing that pages you at three in the morning. And the thing that takes the entire system down with it when it fails. As companies start running fleets of AI agents, that bottleneck gets more expensive, and more fragile, every single year.

## s05_decay  (primitive_decay)

SMESH removes the middle, and replaces it with three mechanisms. The first is decay. Every message carries its own expiry, built into its physics. It weakens along a curve, and when it is weak enough, it is simply gone. Nothing has to clean up. Stale work removes itself.

## s06_reinforce  (primitive_reinforce)

The second is reinforcement. When one agent reaches a conclusion that another agent has already reached, that is not a duplicate to be thrown away. It is a second witness. Confidence rises. Agreement compounds.

## s07_address  (primitive_address)

The third mechanism is the one that makes the other two work. Every claim is addressed by its content — by what is being said, not by who said it. So two agents that independently arrive at the same conclusion land on the exact same address, automatically, without ever having spoken to each other.

## s08_quic  (quic)

Underneath all of it, these agents talk over QUIC — the same encrypted transport that carries modern web traffic. Every connection is peer to peer. Every connection is encrypted. There is no server in the middle relaying anything. There is nothing in the middle to buy, to scale, or to lose.

## s09_setup  (setup)

So let us watch it work. What follows is a recording of an actual run. Five separate programs, on five separate network ports, talking over real encrypted connections. Each one is monitoring the same fleet of services. But each one can only see a single kind of data. One watches response times. One watches errors. One watches capacity. One watches how requests retry. One watches software releases. None of them can see what the others see. And something is about to go wrong.

## s09b_windows  (windows)

Think of them as five people watching the same building through five different windows. One can only see the lobby. One can only see the stairwell. None of them can see the fire. Every one of them sees smoke.

## s10_incident  (demo_establish)

A release goes out. It quietly shrinks a connection pool by ninety percent. And now all five of these watchers see a piece of the damage — but not one of them sees the whole thing. Worse than that: two of them are about to accuse the wrong service entirely.

## s11_mesh  (demo_mesh)

This is the mesh itself. Five agents, six encrypted links. Every dot you see crossing a link is a real message that was actually sent — read back from the recording, not animated for effect. And notice they are not all connected to each other. Some messages have to be passed along by a neighbour to reach the far side, exactly the way a forest relays a signal.

## s12_claims  (demo_claims)

On the right is what the network currently believes. Each row is a claim about one service. The five small tags beneath each claim are the five agents. A tag lights up the moment that agent independently backs the claim. Watch the top row. One agent. Then a second. Then a third — each arriving from completely unrelated evidence, and finding the others already there.

## s13_consensus  (demo_consensus)

There it is. Four independent agents. Four unrelated kinds of evidence. One conclusion. That crosses the threshold, and the network calls it. Nobody voted. Nobody was in charge. The answer assembled itself out of five partial views — and the fifth agent confirms it moments later.

## s14_decoys  (demo_decoys)

Now look at what did not happen. This service was throwing errors loudly. It looks broken. On its own, the error watcher would have blamed it. It collects three witnesses, and it stops there — because it is a symptom, not a cause. And these last three claims only ever found a single witness each. Nothing confirmed them, so they simply fade out. Nobody had to decide they were wrong. Going uncorroborated was enough.

## s15_evidence  (demo_journal)

And every step of it is on the record. Each of those five programs wrote down everything it did, and everything it chose not to do, as it happened. That record is what you have been watching. It is not a reenactment — it is the run itself, played back.

## s16_payoff  (payoff)

That is the whole idea. Not one agent in this run had enough information to be right. The system was right anyway. The telemetry here is synthetic, so the run stays reproducible. The coordination is not synthetic. Those were real processes, on real sockets, making real decisions — recorded, verified, and replayed back to you exactly as they happened.

## s16b_unlocks  (unlocks)

That property is what makes this worth building. Agents can be added without reconfiguring anything. Agents can fail without taking the answer with them. There is no central capacity to outgrow, and no coordinator bill that scales with the fleet. The network gets more reliable as it gets larger, because more witnesses is exactly what it runs on.

## s17_close  (close)

As software moves toward fleets of autonomous agents, the hard problem stops being how clever each agent is. It becomes how they agree. SMESH is a bet that the answer has been running quietly under our feet for four hundred million years.
