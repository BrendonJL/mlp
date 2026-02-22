
---
id: "<% tp.date.now("YYYY-MM-DD") %>"
aliases: []
tags:
  - project/mlp
  - type/daily
  - area/learning
created: <% tp.file.creation_date() %>
---
# 📅 <% tp.date.now("dddd, MMMM DD, YYYY") %>

> _Week <% tp.date.now("WW") %> of <% tp.date.now("YYYY") %>_

## 🎯 Today's Goals

- [ ]
- [ ]
- [ ]

## ✅ What I Accomplished

- [x]  ✅ <% tp.date.now("YYYY-MM-DD") %>

## 🧠 What I Learned

-

## 💡 Challenges & Solutions

- **Challenge**:
- **Solution**:

## 🔜 Tomorrow's Focus

- [ ]  📅 <% tp.date.now("YYYY-MM-DD", 1) %>

## 🔗 Links & Context

- [[ProjectDocumentation]]
- [[<% tp.date.now("YYYY-MM-DD", -1) %>|Yesterday]]
- [[<% tp.date.now("YYYY-MM-DD", 1) %>|Tomorrow]]

## 💻 Code/Commands Used

\```bash

\```

## 📝 Notes

<%*
const hour = tp.date.now("H");
let greeting;
if (hour < 12) greeting = "Good morning!";
else if (hour < 18) greeting = "Good afternoon!";
else greeting = "Good evening!";
%>
_<% greeting %> Start of day <% tp.date.now("h:mm A") %>_
