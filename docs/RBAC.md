# RBAC model

## Roles and clearance

| Role | Clearance | Reads |
|---|:-:|---|
| `admin` | 4 | Everything. Manages users, roles and document access. |
| `product` | 4 | Owns the specifications, including commercial structure. |
| `engineering` | 4 | Builds the product, including the licensing system. |
| `sales` | 4 | Customer guides and the commercial licensing model. |
| `qa` | 3 | Specifications and all test material. Never commercial material. |
| `support` | 3 | Guides and specifications, for product truth on tickets. |
| `customer` | 2 | User and administrator guides only. |

**Clearance is a ceiling, not a grant.** A role with clearance 4 reads nothing
until a `document_acl` row exists for it. Both conditions are required:

```
doc.sensitivity <= role.clearance     AND     role ∈ doc.acl
```

Two independent conditions must both fail before a leak occurs. Clearance
stops a mis-click from handing a Customer a restricted document; the ACL stops
a high-clearance role from reading material that is not theirs.

Engineering and Sales sit at clearance 4 because the License Management BRD is
sensitivity 4 and both legitimately need it. QA and Support are capped at 3, so
no commercial or partner material can ever reach them, even by mistake.

## Sensitivity levels

| | Level | Meaning |
|---|---|---|
| 1 | PUBLIC | Marketing or general material. |
| 2 | CUSTOMER | Operational how-to safe to show paying customers. |
| 3 | INTERNAL | Staff-only product truth: specs, BRDs, test cases, governance packs. Contains roadmap items and known gaps. |
| 4 | RESTRICTED | Commercially sensitive: licensing and partner structure, entitlement envelopes, internal portals, backend runbooks. |

## Starting access matrix

Seed data in `app/db/seed.py`, overridable per document by an admin.

| Document type | L | eng | product | qa | sales | support | customer |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Product & Functional Specification | 3 | Y | Y | Y | – | Y | – |
| Business Requirements Document | 3 | Y | Y | Y | – | Y | – |
| User & Administrator Guide | 2 | Y | Y | Y | Y | Y | Y |
| User Manual | 2 | Y | Y | Y | Y | Y | Y |
| Test Cases | 3 | Y | Y | Y | – | – | – |
| Validation & Governance Pack | 3 | Y | Y | Y | – | – | – |
| **License Management BRD** *(override)* | **4** | Y | Y | – | Y | – | – |
| **License Management User Manual** *(override)* | **3** | Y | Y | – | Y | Y | – |
| *anything unrecognised* | 4 | – | – | – | – | – | – |

`admin` is granted everything and is omitted from the table.

### Why the non-obvious cells look like that

**Sales is denied specifications.** They carry roadmap and deferred items, and
the documents say so themselves — the User Access spec states: *"Roadmap
describes future intent only... the planned audit trail is not to be presented
or tested as available."* A salesperson quoting `UAP-RM-001` as a shipped
feature is a real commercial risk. This is a documented rule, not a guess.

**Support is granted specifications.** Every spec names Support in its own
`Primary audience` field, and answering a ticket needs product truth.

**The License Management BRD is level 4 and excludes QA, Support and
Customer.** It contains partner entitlement envelopes, the commercial model,
internal portal design, and a backend DevOps reduction runbook. QA still gets
the License *Test Cases* at level 3.

**User guides are the only category that reaches Customer.** Without them the
customer-facing half of the product does not exist.

**The License Management User Manual is an override, not a type rule.** It is
titled "User Manual" and the type rule therefore sent it to customers. Its
own masthead reads *"Applies to: AssetCues, partner and customer roles"*,
and it documents the internal subscription portal, the partner allocation
portal and the backend-only Entitlement Reduction path. Classifying by
document type assumes a user manual is written for users; a mixed-audience
document breaks that assumption, and only reading it catches the problem.
The LLM classifier flagged this file as restricted when the deterministic
matrix did not — the strongest argument in this system for keeping both.

**Unrecognised types default to RESTRICTED, admin only.** Default-deny.

## Where the classification comes from

Every AssetCues document carries its own `Primary audience` field:

| Document type | Declared audience |
|---|---|
| Functional Spec / BRD | Product, Engineering, QA, Implementation, Support, Security, Audit |
| User & Administrator Guide | Group/LE Administrators, Application Users, Implementation, Support, Customer Success |
| Validation & Governance Pack | QA, Product, Engineering, Release Management, Audit |

The parser extracts it and the classifier receives it as evidence. It is
strong evidence, not an answer: **no document names "Sales" or "Customer"**,
so routing anything to those roles is a deliberate human decision. The User
Guides are the defensible case, because they name customer-side roles (Group
Admin, Legal Entity Admin, Application User) — which are AssetCues' own
customer personas.

## Per-user overrides

`user_document_grants` holds `ALLOW` and `DENY` rows for one user and one
document, optionally with an expiry.

- **DENY always wins**, over every role grant.
- **ALLOW** grants one document without granting a role — the mechanism for a
  temporary escalation.
- Both are subject to the clearance ceiling and to `status = 'APPROVED'`.

## Tenancy

Every user, document and chunk carries a `tenant_id`. Internal staff live in
the `assetcues` tenant; each customer organisation gets its own. The predicate
requires an exact match, so tenancy is a **hard boundary in both directions**:
Acme cannot read Globex's documents, and internal staff cannot read either
one's. It is not a hierarchy.

## Changing the model

1. Edit `DEFAULT_ROLES` / `DOC_TYPE_ACCESS` in `app/db/seed.py`.
2. Run the suite. `test_no_seeded_grant_exceeds_the_granted_role_clearance`
   fails if a grant would be a dead row, and
   `test_customer_is_never_granted_internal_material` fails if you widen
   customer access past level 2.
3. Re-run `acues-ingest sync` or re-approve affected documents.
