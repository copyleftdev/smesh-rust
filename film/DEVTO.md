---
title: "My QUIC transport had never once been executed. Here's what happened when I ran it."
published: false
description: "I built a plant-inspired coordination protocol, wrote 500 lines of real QUIC networking for it, and never actually turned it on. Wiring it up found three latent bugs in twenty minutes — and then taught me that three of my protocol's core semantics were wrong for real distribution."
tags: rust, distributedsystems, networking, ai
---

I've written before about SMESH, a coordination protocol modelled on mycorrhizal networks — the fungal web that lets trees in a forest warn each other about drought and disease with nothing in charge of the network. Signals diffuse, decay on their own, and get reinforced when independently confirmed. Consensus emerges instead of being orchestrated.

That was the idea. This post is about the part where I found out whether it worked.

## The transport that had never run

SMESH has had a QUIC transport in it for a while. Roughly 500 lines: a quinn endpoint that is simultaneously server and client, self-signed certs, length-prefixed bincode frames over unidirectional streams, an accept loop that spawns per-connection and per-stream tasks, connection pooling.

Every test passed. The workspace was green. I could point at `smesh-runtime/src/transport.rs` and say "yes, it does peer-to-peer."

Then I grepped for who actually constructed it:

```
$ grep -rn "QuicTransport" --include='*.rs' .
smesh-runtime/src/transport.rs:177:pub struct QuicTransport {
smesh-runtime/src/transport.rs:192:impl QuicTransport {
smesh-runtime/src/lib.rs:16:pub use transport::{QuicTransport, ...};
```

Its own definition, and a re-export. Nothing else in the workspace had ever instantiated it. No binary opened a socket. `SmeshRuntime` imported `TransportConfig`, stored it in a struct field, and never looked at it again.

I had a networking layer with tests, docs, and zero executions.

## Three bugs in the first twenty minutes

I wrote an integration test that starts two runtimes, has one dial the other, and asserts a signal crosses. Here is what fell out before it went green.

**1. It panicked on the first call.**

```
Could not automatically determine the process-level CryptoProvider
from Rustls crate features.
```

rustls 0.23 refuses to pick a crypto backend when more than one is compiled in, and quinn pulls in both through its own feature set. Every call to `QuicTransport::new` would have panicked for anyone, ever. Nobody noticed because nobody had called it.

**2. Dialled connections were write-only.**

`connect()` stored the connection in the pool but only the accept loop pumped incoming streams — and the accept loop only sees connections you *accepted*. So a node that dialled out could send, and would never receive anything back. A QUIC connection is bidirectional regardless of who dialled it; my code only acted like it half the time.

**3. Unbounded allocation from an attacker-controlled length prefix.**

```rust
let len = u32::from_be_bytes(len_buf) as usize;
let mut data = vec![0u8; len];   // <- no
```

`max_message_size` was in the config struct. It was never read. Send a 4 GiB length prefix and the process allocates 4 GiB.

None of these are clever bugs. They're the bugs you get for free the first time code meets a socket, and the only reason they survived is that the code had never met a socket.

## The harder problem: my protocol was wrong

Fixing the plumbing was the easy half. The real issue was that my diffusion algorithm quietly assumed something no distributed system can assume.

`Network::tick` expands a signal's reach one hop per tick by walking the graph:

```rust
for node_id in &reached {
    for hypha in self.hyphae.get(node_id) {
        let target = self.nodes.get(&hypha.to);
        if target.should_relay(&signal, remaining_hops) {
            frontier.push(hypha.to.clone());
        }
    }
}
```

Read that again. It iterates every node's adjacency, calls every node's relay policy, and mutates one global `reached_nodes` set on a shared signal. It's a breadth-first search from a god's-eye view of the entire graph.

That works beautifully in one process. In a real mesh, **no node can see that graph.** Porting it meant three corrections, and each one turned out to be a genuine bug rather than a porting detail.

### Correction 1: independent conclusions were being thrown away

Signals are content-addressed — the hash is derived from what is being claimed. So when two agents independently reach the same conclusion, they produce the same hash.

My `emit()` saw the hash already present locally and treated it as a duplicate. It dropped it.

That is exactly backwards. Two parties independently agreeing is not redundant data — it is the *only* evidence the system has that a claim is real. Discarding it destroys the thing the protocol exists to measure.

```rust
/// Signals are content-addressed, so a node that independently reaches a
/// conclusion another node already published lands on the same hash. That
/// is treated as *corroboration*: this node is added as an attester and the
/// merged claim still goes out, because our agreement is news to everyone
/// who has not heard it. Swallowing it as a duplicate would silently
/// discard the only evidence that two parties concur.
```

### Correction 2: I was counting messengers, not witnesses

Reinforcement attributed the claim to whoever handed me the message. In a gossip mesh, one finding relayed by five nodes then looks like five corroborators.

The party that attests to a claim is its *origin*, not the peer that passed it along. Relaying is not agreeing.

### Correction 3: gossip needs a merge rule, not a broadcast rule

Once attestation is a set, the rule that makes gossip converge is simple and pleasant:

```rust
// Merge the two attester sets. Anything the sender knew that we did
// not is new information, and new information is worth passing on.
let before = Node::attesters(existing);
for attester in &incoming_attesters {
    if !before.contains(attester) {
        existing.reinforce(attester);
    }
}
```

Forward if and only if your own knowledge grew. That single rule is the loop breaker (a message teaching you nothing goes no further), the convergence mechanism (the set is grow-only, so it's a CRDT), and the anti-entropy repair (re-asserting carries your accumulated view, so a node that missed a round catches up).

## The part I nearly got wrong on purpose

Here is the subtlety that makes the whole thing work, and it looks like a mistake in the source.

The signal builder folds the origin node into the content hash if you give it one. So when an agent publishes a claim, you must **not** set the origin:

```rust
let signal = Signal::builder(SignalType::Alert)
    .payload(assertion.canonical_bytes())
    .confidence(finding.confidence)
    .build();
// .origin() deliberately NOT called
```

If you set it, every agent gets a different hash for the same claim and correlation becomes impossible. The address has to be the *claim*, not the *claimant*.

The corollary is that evidence cannot travel in the payload. Each agent's evidence differs, so putting it in would make every hash unique and break the mechanism. The payload is just the assertion:

```json
{"subject":"checkout-api","claim":"degraded"}
```

Evidence stays local and goes to the log. The mesh carries assertions; it does not carry arguments.

## Five witnesses, none of whom can see the problem

To actually test any of this, I built a scenario where the answer cannot be reached alone.

Five analyst processes watch the same fleet of services. Each can see exactly one kind of telemetry, and nothing else:

| agent | sees |
| --- | --- |
| latency | p99 response times |
| errors | error rates |
| saturation | pool and CPU utilisation |
| traces | retry rates, span queueing |
| deploys | release events |

A deploy cuts `checkout-api`'s connection pool from 200 to 20. The pool pins, requests queue, callers time out.

Now every agent sees a piece of it, and two of them are actively misleading. `errors` sees `payments-api` throwing 503s and would blame it — but payments is a victim, its own pool and CPU are fine. `latency` sees four services slow at once and can't say which is causal. Only `deploys` can see there was a release, and a release on its own means nothing; software ships all day.

I also planted two decoys with a single witness each — an unrelated CPU spike, a brief latency blip — because a system that agrees with everything is not consensus, it's an echo.

They run as five real OS processes on five ports over real encrypted QUIC, in a ring-plus-chord topology so messages actually have to be relayed to cross the mesh. Peer discovery is off, or gossip quietly converts any topology into a full mesh and there's nothing left to watch.

```
$ smesh orchestrate --out runs/latest

  spawned latency     pid 2452744  127.0.0.1:9301  dials 9302,9303
  spawned errors      pid 2452745  127.0.0.1:9302  dials 9303
  spawned saturation  pid 2452746  127.0.0.1:9303  dials 9304
  spawned traces      pid 2452747  127.0.0.1:9304  dials 9305
  spawned deploys     pid 2452748  127.0.0.1:9305  dials 9301
```

The result:

```
what the mesh concluded
  checkout-api          CONSENSUS at 16.9s · 5 attesters · seen by 5 nodes
  payments-api          no consensus (3/4 concerns)
  edge-gateway          no consensus (1/4 concerns)
  notification-worker   no consensus (1/4 concerns)
  session-store         no consensus (1/4 concerns)
```

Cause separated from symptom separated from noise. The loud, obvious suspect was held at three witnesses and never promoted. The decoys were never rejected by anything — they simply went uncorroborated and decayed out. Nobody voted, and nothing was in charge.

## Making the run replayable, and then checking that claim

I wanted to visualise this, which meant the log had to be good enough to reconstruct the run exactly. Each process writes newline-delimited JSON against a shared run epoch: every emission, every per-peer send, every receipt, every relay decision including the probability and the die roll that resolved it, and a full field snapshot every 500ms so decay curves are *observed* rather than modelled.

Then I did the part I'd recommend to anyone building a log you intend to trust: I wrote a validator that checks the log against itself. Sequence gaps, time going backwards, snapshots referencing signals that were never received, consensus declared without the receipts to justify it.

It caught a real bug on its first run:

```
FAIL  latency: first event is peer_connected, not node_started
```

A peer completed the handshake in the gap between binding the endpoint and writing the node's identity line. The fix was to move the identity write inside the mesh startup, before any loop spawns. I would never have found that by looking at the picture — the picture would just have been subtly wrong.

## The hole this left open

Writing the above, I had to be honest that the headline claim was not yet true.

"Five independent agents corroborate this" was measured by counting **strings**. `origin_node_id` was a bare name on the wire, and the reinforcement list was just more names. Any single node could append four of them and manufacture unanimous agreement for a claim nobody else had ever seen. The protocol's central measurement was forgeable by one participant.

So attestations are now signatures. Each is an Ed25519 signature over the claim's content hash, bound to the attester's own name:

```rust
fn attestation_message(claim_hash: &str, node_id: &str) -> Vec<u8> {
    let mut message = Vec::new();
    message.extend_from_slice(claim_hash.as_bytes());
    message.push(0x1f);  // separator, so (a,bc) and (ab,c) cannot collide
    message.extend_from_slice(node_id.as_bytes());
    message
}
```

Binding the name into the signed bytes is what stops an attestation being replayed under a different name. Counting attesters is now counting signatures, and unverifiable ones are dropped rather than counted.

Signatures prove key ownership, not *name* ownership — nothing there stops a peer calling itself `latency`. The mesh closes that separately by pinning a name to the key that first presented it, and refusing later keys for that name. Trust on first use: no help if the impostor arrives first, but the name is unstealable for the rest of the run.

Three tests carry the property, and they are the ones I would read first:

- a claim nobody signed never enters the field
- a real signature lifted onto a different claim does not verify
- two nodes independently reaching the same conclusion produce two signatures on one signal

The content hash went from 64 bits to 128 in the same change. 64 was fine against accident, but signatures are now taken *over* that hash, so a collision would let agreement on one claim be presented as agreement on another.

## The NAT question, and what I could not prove

I assumed NAT blocked participation. Testing it showed otherwise: a node that can only dial *outward* participates fully, because a QUIC connection is bidirectional regardless of who opened it. Signals flow back over the connection the NATed node itself established.

What NAT actually breaks is narrower. A node behind one cannot be *discovered* — and here there was a real defect: peer gossip shared the address a peer said it was listening on, which behind NAT is a private address no third party can route to. We were handing out routes that could never work.

That is fixed by carrying candidates rather than one address: the local one, and the one a peer reports actually seeing traffic arrive from. The second is discovered the way STUN does it, except a peer supplies it instead of a server:

```
learned our address is 192.168.0.35:9971 (per observer)
```

The node had bound `0.0.0.0:9971`. It had no way to know that itself.

Two peers reporting *different* addresses for you means the translator allocates a fresh mapping per destination — symmetric NAT — and no amount of address sharing will help. The code warns rather than failing to connect later for no visible reason.

The remaining case is two nodes both behind NAT, which needs a simultaneous open coordinated by someone they can both already reach. That is implemented and the coordination path is tested end to end.

So I put it on three cloud hosts in three regions — two of them inside network namespaces behind real `MASQUERADE` NATs, and a public rendezvous — and watched what happened. It found three bugs.

**We advertised `0.0.0.0`.** A node bound to every interface reports exactly that as the address it listens on, and it was going out as a candidate. The other side dutifully tried to dial it and burned a connect timeout on an address that cannot answer.

**One slow dial stalled every other message.** The inbound loop handled discovery inline, so a five-second connect attempt blocked the socket reader — including the reply carrying our own public address. The result was a node asking to be punched to *before it knew where it was*, advertising the useless address above. The logs are unambiguous:

```
05:41:11.955  trying left at 138.197.31.115:9401
05:41:16.957  learned our address is 147.182.229.187:9402   <- five seconds later
```

**The simultaneous open was not simultaneous.** The requester dialled, failed, *then* asked the relay to tell the other side to dial. The two attempts ended up a full timeout apart and never overlapped — which is the entire mechanism. Both sides now dial at once, repeatedly.

Then the interesting part. With all three fixed, both ends were dialling each other's correct public addresses in the same window, and it still did not connect. The packet capture said why:

```
left -> rendezvous  :  138.197.31.115:9401
left -> right       :  138.197.31.115:50343
```

A different external port per destination. That is symmetric NAT, and no amount of address sharing survives it: right was told to expect `:9401` and left arrives from `:50343`, so both directions get dropped. Hole punching cannot work through it, which is exactly what the code already said it could not do — now measured rather than assumed.

## The part that made it not matter

Here is the thing I would have missed by reasoning instead of testing. A signal emitted by the node behind the New York NAT arrived at the node behind the San Francisco NAT:

```
[recv] 155ad5b1e52d48b6ba93daf18fdb501b from rendezvous (hop 1)
```

Hop one. Relayed through a peer both of them could reach.

The mesh never needed the direct connection. Relaying through intermediate peers is what a gossip protocol does anyway, so two nodes that cannot possibly connect to each other still coordinate. Hole punching is an optimisation that saves a hop; it is not a prerequisite for participation.

That reframes the whole NAT question. The thing worth engineering was never traversal. It was making sure a node with nothing but outbound connectivity is a full participant — and it already was.

## What is still wrong

- **Hole punching works for cone NATs and not symmetric ones.** The cone case is still untested; the symmetric failure is measured. Nodes behind symmetric NAT fall back to relaying, which costs a hop and a little latency.
- **The telemetry in the demo is synthetic.** Deliberately: a seeded fixture means the run reproduces byte-for-byte on any machine, which is what makes a visualisation worth trusting. The coordination is not synthetic — real processes, real sockets, probabilistic relay.
- **Trust on first use is not identity.** There is no key distribution and no revocation. A node that generates its own name rather than deriving it from its key is only as trustworthy as whoever it met first.
- **The recording predates the signing work.** The run in the video was captured before attestations were signatures, so what you are watching is the mechanism, not the hardened version of it.

## See it move

The full narrated walkthrough is the cover video on this post, or here: https://youtu.be/kmCzwSBqu_s

It opens on the forest the protocol is stolen from, then goes inside the recorded run: five agents, six encrypted links, every dot on screen a real message read back from the log rather than animated for effect.

If you take one thing from this: **code that has never been executed is not code, it's a plan.** Mine had tests, docs, and a clean `cargo clippy`, and it would have panicked on the first line for every user. The tests were testing that the plan was internally consistent.

Repo: [github.com/copyleftdev/smesh-rust](https://github.com/copyleftdev/smesh-rust) · Rust · MIT/Apache-2.0
